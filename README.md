# RWallpprs `v0.2.0-beta.1` — Видео и Нативные Rust-обои под Niri (Arch Linux)

**RWallpprs** — движок живых и процедурных обоев для скроллингового тайлингового Wayland-композитора **Niri** на базе **Arch Linux**.

## Что изменено в `v0.2.0-beta.1` (устранение детских болезней концепта)

1. **Нативный мультимедиа-стек через `ffmpeg-next` + `ffmpeg-sys-next` (без CLI-костылей `Command::new("ffmpeg")`)**:
   - В `rwallpprs-daemon` интегрированы крейты `ffmpeg-next` и `ffmpeg-sys-next` (`libavformat`, `libavcodec`, `libswscale`, `libavutil`).
   - Аппаратный контекст декодирования (`AV_HWDEVICE_TYPE_VAAPI`, `AV_HWDEVICE_TYPE_VULKAN`, `AV_HWDEVICE_TYPE_CUDA`, `AV_HWDEVICE_TYPE_VDPAU`) поднимается внутри процесса демона через `av_hwdevice_ctx_create` с привязкой к `AVCodecContext::hw_device_ctx`.
   - Аппаратные кадры (`AV_PIX_FMT_VAAPI` / `VULKAN` / `CUDA`) переносятся через `av_hwframe_transfer_data` в преаллоцированный `frame::Video` и конвертируются через `sws_scale` (`software::scaling::Context`) напрямую в маппированный слот `wl_shm` (`XRGB8888`).
   - Бесшовный цикл видео (`ictx.seek(0, ..0)` + `decoder.flush()`) без пересоздания пайплайна и без лишних аллокаций на кадр.
2. **Официальный крейт `niri-ipc` вместо ручного парсинга сокета и хардкода разрешений**:
   - Модуль `niri_monitor.rs` использует типы `niri_ipc::socket::Socket`, `niri_ipc::Request::EventStream`, `niri_ipc::Event`, `niri_ipc::Window`, `niri_ipc::Workspace`.
   - Никаких захардкоженных проверок `1900x1060`: перекрытие экрана (`active_column_maximized`) рассчитывается динамически по отношению `tile_size` активного окна к реальному логическому разрешению монитора (`output_width`, `output_height`), полученному от `zwlr_layer_surface_v1::configure`.
3. **Аппаратный GPU-бэкенд `wgpu` (Vulkan/GL) в `rwallpprs-sdk`**:
   - В `rwallpprs-sdk` добавлен полноценный headless-рендерер на базе `wgpu` (`WgpuShaderRunner` и типаж `GpuShaderWallpaper`).
   - Обои на Rust теперь можно писать двумя способами:
     - **На GPU (WGSL-шейдеры через Vulkan/`wgpu`)** с автоматической передачей uniform-буфера (`time`, `resolution`, `niri_scroll_offset`, `niri_workspace_idx`, `niri_overview`) и выгрузкой кадра через выровненный по `COPY_BYTES_PER_ROW_ALIGNMENT` (256 байт) staging-буфер прямо в разделяемую память `memfd`.
     - **На CPU (`Canvas`)** с прямым доступом к срезу `&mut [u32]` в `memfd`.
4. **Безопасные системные вызовы через `rustix` и персистентный `memfd` для вложенных бинарников**:
   - Сырые вызовы `libc::memfd_create`, `libc::mmap`, `libc::ftruncate` заменены на типизированные обертки `rustix::fs` и `rustix::mm` с дескрипторами `OwnedFd`.
   - При вызове `rwallpprs run-embedded` CLI больше не использует хрупкий `sleep(250ms)` со ссылкой на свой `/proc/<CLI_PID>/fd/<FD>`. Вместо этого ELF-образ передается демону по сокету (`ControlCommand::SetEmbeddedBin`), демон сам создает долгоживущий `memfd` в своем адресном пространстве и запускает бинарник через `/proc/self/fd/<fd>`, сохраняя способность перезапускать его при смене разрешения монитора.
5. **Zero-copy direct present без единого memcpy на кадр**:
   - `wl_shm`-пул демона теперь создается **поверх того же `memfd`, куда рендерит движок** (Rust-бинарник через SHM-протокол или ffmpeg-декодер через собственное memfd-кольцо демона): `wlr_layer_shell`-поверхность отображает кадры через `attach` прямых `wl_buffer` (`DirectShmPresent` в `wayland_layer.rs`), и слоты чередуются 0/1 по `wl_buffer.release`.
   - При смене разрешения монитора или недоступности direct-режима основной цикл прозрачно откатывается на прежний копирующий путь (`present_frame` + `SlotPool`), поэтому визуальный вывод деградирует без обрыва анимации.

---

## Сборка и запуск на Arch Linux

```bash
# 1. Установка системных библиотек Arch Linux
sudo pacman -S --needed base-devel rust clang pkgconf ffmpeg libva libva-mesa-driver vulkan-icd-loader wayland wayland-protocols

# 2. Сборка воркспейса
cargo build --release

# 3. Запуск демона
./target/release/rwallpprs daemon --config config/rwallpprs.toml &

# 4. Запуск GPU/CPU Rust-обоев (с реакцией на скролл ленты Niri)
./target/release/rwallpprs set-bin ./target/release/niri-cyber-grid --mode animated --fps 60

# 5. Встраивание бинарника обоев внутрь CLI и запуск из памяти демона
./target/release/rwallpprs bundle-bin ./target/release/niri-voronoi-flow -o ./rwallpprs-voronoi
./rwallpprs-voronoi run-embedded --mode animated --fps 60

# 6. Запуск видео-обоев (H.264 / HEVC / VP9 / AV1) через нативный ffmpeg-next + VA-API
./target/release/rwallpprs set-video ~/Videos/wallpaper_av1.mp4 --hw-accel vaapi --fps 60

# 7. Порт экранной заставки Android 17 Easter Egg «Landroid» (examples/niri-landroid):
#    корабль летает по вселенной на автопилоте (полностью на GPU/wgpu).
./target/release/rwallpprs set-bin ./target/release/niri-landroid --mode animated --fps 60
```

## Пример `niri-landroid` (порт Android 17 Easter Egg «Landroid»)

`examples/niri-landroid` — полный порт экранной заставки (DreamService) Android 17 Easter Egg:
корабль Landroid летает по процедурной вселенной на автопилоте, камера следует за ним
(dynamic zoom), как в `DreamUniverse`/`MainActivity.Spaaaaace`. Перенесены и симуляция
(`Physics.kt`, `Universe.kt`, `Autopilot.kt`: gravity, посадки/посадочные констрейнты, трек,
искры выхлопа, ringfence), и рендер `VisibleUniverse.kt` (поля гравитации, венцы звезды,
диски/обводки/текстуры планет, орбиты, пунктирный `Container`, флаг посадки, оверлей
автопилота, трек). Вся графика — один инстансный SDF-пайплайн `wgpu` + CPU-триангулированный
полигон корабля.

Сид вселенной задаётся так (порядок приоритета): аргумент `--seed N` → переменная окружения
`RWALLPPRS_SEED=N` → dailySeed (`год*10000 + месяц*100 + день` по UTC, как `randomSeed()` в
оригинале). Один и тот же сид даёт бит-в-бит одинаковую вселенную, что удобно для проверок:

```bash
./target/release/niri-landroid --seed 12345                     # обычный режим (его запускает демон через SHM)
RWALLPPRS_SELFTEST=1200 RWALLPPRS_DUMP_PPM=/tmp/f.ppm \
    ./target/release/niri-landroid --seed 12345                 # self-test: N кадров + PPM-дамп
RWALLPPRS_STATS=1 ./target/release/niri-landroid --seed 12345    # телеметрия по кадрам в stderr
RWALLPPRS_DEBUG_SHIP_R=190000 ./target/release/niri-landroid --seed 12345  # корабль у ringfence (диагностика)
```

**Точность и производительность.** Рендер сверен с исходниками кадра за кадром: порядок
отрисовки (сетка → constraints → звезда → планеты/искры → автопилот → корабль → трек),
штриховка линий в экранных пикселях (`1/zoom`, `2/zoom`, `3/zoom`), пунктир ringfence 8/8
экранных px независимо от зума, формулы полей гравитации, венцов звезды, текстур планет и
оверлея автопилота. Невидимые объекты, искры и сегменты трека отсекаются по расширенному на
10 % кадру — отсечение консервативно (тонкие окружности и огромные кольца проверяются на
пересечение с кадром, а не по своему центру), поэтому картинка не меняется (проверено
попиксельным диффом с отсечением и без: 0 различающихся пикселей на разных сценах), а время
кадра падает примерно вдвое: на 1920×1080 в self-test ~3.3 мс/кадр с отсечением против
~6.5 мс/кадр без него (медиана из 3 парных прогонов по 1200 кадров, Vulkan; абсолютные
значения зависят от нагрузки на GPU).
