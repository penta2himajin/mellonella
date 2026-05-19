//! `eframe::App` implementation — the egui UI rendering loop.
//!
//! Single-window layout, minimal styling. Tray-side menu events
//! (Show / Start / Stop / Quit) come in via the channel the
//! `tray` module installs; we drain it on each `update()` call.

use eframe::egui;

use crate::state::{AppState, EnrollmentOrigin, OUTPUT_SAMPLE_RATE};
use crate::tray::{TrayCommand, TrayHandles};

pub struct MellonellaApp {
    state: AppState,
    tray: Option<TrayHandles>,
    /// Whether the main window is currently visible. Tracked
    /// separately so the tray menu's "Show / Hide" can flip it
    /// without going through the OS close request loop.
    window_visible: bool,
}

impl MellonellaApp {
    pub fn new(state: AppState, tray: Option<TrayHandles>) -> Self {
        Self {
            state,
            tray,
            window_visible: true,
        }
    }

    fn drain_tray_commands(&mut self, ctx: &egui::Context) {
        let Some(tray) = self.tray.as_ref() else {
            return;
        };
        while let Some(cmd) = tray.try_recv() {
            match cmd {
                TrayCommand::Show => {
                    self.window_visible = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Visible(true));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
                }
                TrayCommand::Toggle => {
                    if self.state.is_running() {
                        self.state.stop();
                    } else {
                        self.state.start();
                    }
                }
                TrayCommand::Quit => {
                    self.state.stop();
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
    }

    fn render_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Mellonella");
            ui.add_space(8.0);

            self.render_enrollment_section(ui);
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(6.0);
            self.render_device_row(ui);
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(8.0);
            self.render_run_controls(ui);
            ui.add_space(8.0);
            self.render_meters(ui);
            ui.add_space(6.0);
            self.render_status(ui);
            ui.add_space(6.0);
            self.render_error_row(ui);
            ui.add_space(6.0);
            self.render_settings_panel(ui);
        });
    }

    /// Top-of-window enrollment summary:
    /// - During an active recording: progress bar
    /// - No pool loaded: big "welcome" wizard panel asking the user
    ///   to record their voice (first-run experience)
    /// - Pool loaded: compact "● Profile" pill + Re-enroll button
    fn render_enrollment_section(&mut self, ui: &mut egui::Ui) {
        if self.state.is_recording() {
            ui.label("Enrollment:");
            self.render_recording_progress(ui);
            return;
        }
        if self.state.pool.is_some() {
            self.render_profile_pill(ui);
        } else {
            self.render_first_run_wizard(ui);
        }
    }

    /// Compact "● Profile loaded" indicator + Re-enroll button. The
    /// power-user "From WAV / Load / Save JSON" buttons live in the
    /// Settings panel; the main UI exposes only the action the user
    /// would actually want from here.
    fn render_profile_pill(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let origin_short = match &self.state.origin {
                EnrollmentOrigin::None => "voice".to_string(),
                EnrollmentOrigin::Wav(_) => "voice (from WAV)".to_string(),
                EnrollmentOrigin::Mic { secs } => format!("voice ({secs}s recording)"),
                EnrollmentOrigin::Json(_) => "voice (from JSON)".to_string(),
            };
            ui.label(
                egui::RichText::new(format!("● Profile: {origin_short}"))
                    .color(egui::Color32::from_rgb(80, 200, 120)),
            );
            ui.label(format!("· {} anchors", self.state.pool_anchors));
            if self.state.pool_f0_mu > 0.0 {
                ui.weak(format!(
                    "· F0 μ={:.0} Hz σ={:.0} Hz",
                    self.state.pool_f0_mu, self.state.pool_f0_sigma
                ));
            }
            // Re-enroll button lives at the right-hand edge of the row.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let busy = self.state.is_running();
                let record_secs = self.state.record_duration_secs;
                if ui
                    .add_enabled(!busy, egui::Button::new("Re-enroll"))
                    .on_hover_text("Replace the saved profile with a new mic recording.")
                    .clicked()
                {
                    self.state.start_recording(record_secs);
                }
            });
        });
    }

    /// First-run welcome / set-up-your-voice panel. Shown when no pool
    /// has been loaded yet (either because the user just installed the
    /// app, or they cleared `~/.config/mellonella/enrollment.json`).
    fn render_first_run_wizard(&mut self, ui: &mut egui::Ui) {
        egui::Frame::new()
            .inner_margin(egui::Margin::same(16))
            .fill(ui.visuals().faint_bg_color)
            .corner_radius(egui::CornerRadius::same(6))
            .show(ui, |ui| {
                ui.label(egui::RichText::new("Welcome to Mellonella").heading());
                ui.add_space(4.0);
                ui.label(
                    "Mellonella filters live audio so only the enrolled speaker's voice \
                     passes through. Record a short sample of your own voice to begin — \
                     the rest is automatic.",
                );
                ui.add_space(12.0);
                let record_secs = self.state.record_duration_secs;
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                egui::RichText::new(format!(
                                    "🎙  Record my voice ({record_secs:.0} s)"
                                ))
                                .strong(),
                            )
                            .min_size(egui::vec2(220.0, 36.0)),
                        )
                        .clicked()
                    {
                        self.state.start_recording(record_secs);
                    }
                    ui.add(
                        egui::DragValue::new(&mut self.state.record_duration_secs)
                            .speed(0.5)
                            .range(1.0..=30.0)
                            .suffix(" s")
                            .fixed_decimals(0),
                    )
                    .on_hover_text("Recording duration (1 – 30 s)");
                });
                ui.add_space(4.0);
                ui.weak(
                    "Tip: speak naturally for the duration. The file is saved to \
                     ~/.config/mellonella/enrollment.json and reused on every launch.",
                );
            });
    }

    fn render_recording_progress(&mut self, ui: &mut egui::Ui) {
        let (elapsed, target, progress) = self
            .state
            .recorder
            .as_ref()
            .map_or((0.0, self.state.record_duration_secs, 0.0), |r| {
                (r.elapsed_seconds(), r.target_seconds(), r.progress())
            });
        ui.horizontal(|ui| {
            ui.label(format!("Recording {elapsed:.1} / {target:.1} s"));
            ui.add(
                egui::ProgressBar::new(progress)
                    .desired_width(120.0)
                    .desired_height(14.0),
            );
            if ui.button("Cancel").clicked() {
                self.state.cancel_recording();
            }
        });
    }

    /// Power-user enrollment IO buttons — From WAV, Load JSON, Save
    /// JSON. Hidden from the main page (those buttons are noise for
    /// the common case where the user just records once via the
    /// first-run wizard) and surfaced here for CLI interop.
    fn render_enrollment_power_user_actions(&mut self, ui: &mut egui::Ui) {
        let busy = self.state.is_running() || self.state.is_recording();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("Enroll from WAV…"))
                .on_hover_text("Pick a clean voice WAV (16-bit signed mono, any rate).")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("WAV audio", &["wav"])
                    .pick_file()
                {
                    self.state.enroll_from_wav(&path);
                }
            }
            if ui
                .add_enabled(!busy, egui::Button::new("Load JSON…"))
                .on_hover_text("Load a pre-computed enrollment.json from the CLI.")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Enrollment JSON", &["json"])
                    .pick_file()
                {
                    self.state.load_enrollment_json(&path);
                }
            }
            let has_pool = self.state.pool.is_some();
            if ui
                .add_enabled(!busy && has_pool, egui::Button::new("Save JSON…"))
                .on_hover_text("Export the current profile (in addition to the auto-save).")
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Enrollment JSON", &["json"])
                    .set_file_name("enrollment.json")
                    .save_file()
                {
                    self.state.save_enrollment_json(&path);
                }
            }
        });
    }

    fn render_device_row(&mut self, ui: &mut egui::Ui) {
        let busy = self.state.is_running() || self.state.is_recording();
        ui.horizontal(|ui| {
            ui.label("Input:");
            let current = self
                .state
                .selected_input
                .clone()
                .unwrap_or_else(|| "(default)".into());
            ui.add_enabled_ui(!busy, |ui| {
                egui::ComboBox::from_id_salt("input_combo")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.state.selected_input, None, "(default)");
                        for d in &self.state.available_inputs {
                            ui.selectable_value(
                                &mut self.state.selected_input,
                                Some(d.name.clone()),
                                &d.name,
                            );
                        }
                    });
            });
        });
        ui.horizontal(|ui| {
            ui.label("Output:");
            let current = self
                .state
                .selected_output
                .clone()
                .unwrap_or_else(|| "(default)".into());
            ui.add_enabled_ui(!busy, |ui| {
                egui::ComboBox::from_id_salt("output_combo")
                    .selected_text(current)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.state.selected_output, None, "(default)");
                        for d in &self.state.available_outputs {
                            ui.selectable_value(
                                &mut self.state.selected_output,
                                Some(d.name.clone()),
                                &d.name,
                            );
                        }
                    });
            });
            if ui
                .add_enabled(!busy, egui::Button::new("Refresh"))
                .clicked()
            {
                self.state.refresh_devices();
            }
        });
    }

    /// Pause / Resume toggle. The app auto-starts on first launch
    /// when `can_start()` is true (see [`Self::maybe_auto_start`]);
    /// this button just lets the user temporarily release the mic /
    /// stop filtering without exiting the app.
    fn render_run_controls(&mut self, ui: &mut egui::Ui) {
        let running = self.state.is_running();
        let can_start = self.state.can_start();
        ui.horizontal(|ui| {
            if running {
                if ui
                    .add(egui::Button::new("⏸ Pause").min_size(egui::vec2(140.0, 32.0)))
                    .clicked()
                {
                    self.state.user_paused = true;
                    self.state.stop();
                }
            } else {
                let label = if self.state.user_paused {
                    "▶ Resume"
                } else {
                    "▶ Start"
                };
                if ui
                    .add_enabled(
                        can_start,
                        egui::Button::new(label).min_size(egui::vec2(140.0, 32.0)),
                    )
                    .clicked()
                {
                    self.state.user_paused = false;
                    self.state.start();
                }
            }
            let status = if running {
                egui::RichText::new("● filtering").color(egui::Color32::from_rgb(80, 200, 120))
            } else if self.state.user_paused {
                egui::RichText::new("⏸ paused").color(egui::Color32::from_rgb(220, 200, 80))
            } else if can_start {
                egui::RichText::new("○ starting…").color(egui::Color32::from_rgb(220, 200, 80))
            } else {
                egui::RichText::new("○ enroll a voice first").color(egui::Color32::DARK_GRAY)
            };
            ui.label(status);
        });
        let dfn3_available = self.state.dfn3_available();
        let tse_available = self.state.tse_available();
        ui.horizontal(|ui| {
            ui.weak(if dfn3_available {
                "● noise suppression"
            } else {
                "○ noise suppression (MELLONELLA_DFN3_ONNX not set)"
            });
            ui.separator();
            ui.weak(if tse_available {
                "● target-speaker extraction"
            } else {
                "○ target-speaker extraction (pick ONNX in Settings)"
            });
        });
    }

    /// Auto-start a session on the first frame where conditions allow
    /// it. Called from [`eframe::App::update`] before rendering so the
    /// UI sees the running state immediately. Honours `user_paused` —
    /// a manual Pause click stays paused until the user clicks Resume.
    /// On construction failure (missing ONNX env var, etc.) flips
    /// `user_paused` so the loop doesn't spam the failure on every
    /// frame; the user can investigate the error and click Resume to
    /// retry once it's fixed.
    fn maybe_auto_start(&mut self) {
        if self.state.user_paused || self.state.is_running() || !self.state.can_start() {
            return;
        }
        self.state.start();
        if self.state.last_error.is_some() {
            self.state.user_paused = true;
        }
    }

    fn render_status(&self, ui: &mut egui::Ui) {
        let s = self.state.last_stats;
        let secs = s.samples_processed as f32 / OUTPUT_SAMPLE_RATE as f32;
        let latency_ms = self.state.estimated_latency_ms();
        ui.label(format!(
            "Processed: {secs:.1} s   ·   latency: ~{latency_ms:.0} ms   ·   overruns: {}   underruns: {}",
            s.input_overruns, s.output_underruns
        ));
    }

    /// Step 18: live level meter + gate light. Renders two thin
    /// progress bars (input RMS, output RMS) and a coloured circle
    /// for the gate state. RMS is mapped via a log-ish scale so
    /// quiet speech (~-30 dBFS) still shows movement.
    fn render_meters(&self, ui: &mut egui::Ui) {
        let running = self.state.is_running();
        let in_rms = self.state.input_rms();
        let out_rms = self.state.output_rms();
        let gate_on = self.state.gate_on();

        ui.horizontal(|ui| {
            ui.label("Mic:");
            ui.add(
                egui::ProgressBar::new(rms_to_bar(in_rms))
                    .desired_width(160.0)
                    .desired_height(10.0)
                    .fill(egui::Color32::from_rgb(80, 160, 220)),
            );
            ui.label("Out:");
            ui.add(
                egui::ProgressBar::new(rms_to_bar(out_rms))
                    .desired_width(160.0)
                    .desired_height(10.0)
                    .fill(egui::Color32::from_rgb(180, 180, 80)),
            );
            let (gate_label, gate_colour) = if !running {
                ("○ off", egui::Color32::DARK_GRAY)
            } else if gate_on {
                ("● gate ON", egui::Color32::from_rgb(80, 200, 120))
            } else {
                ("○ gate", egui::Color32::from_rgb(160, 80, 80))
            };
            ui.label(egui::RichText::new(gate_label).color(gate_colour));
        });
    }

    /// Step 19: collapsible "Settings" section with sliders for
    /// the user-tunable gate / envelope / refresh-cadence
    /// parameters. Disabled while a session is running — the
    /// streaming pipeline reads the config at `LiveSession::new`,
    /// not per-frame, so mid-stream changes wouldn't take effect
    /// until Stop / Start anyway.
    #[allow(clippy::too_many_lines)]
    fn render_settings_panel(&mut self, ui: &mut egui::Ui) {
        let running = self.state.is_running();
        egui::CollapsingHeader::new("Settings")
            .default_open(false)
            .show(ui, |ui| {
                ui.add_enabled_ui(!running, |ui| {
                    ui.label(
                        egui::RichText::new("Tunable parameters (applied on next Start)")
                            .small()
                            .color(egui::Color32::GRAY),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.state.gate_cfg.theta_pass, 0.0..=1.0)
                            .text("theta_pass (gate threshold)")
                            .fixed_decimals(2),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.state.gate_cfg.hangover_ms, 0.0..=1000.0)
                            .text("hangover_ms")
                            .fixed_decimals(0),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.state.gate_cfg.attack_ms, 0.0..=100.0)
                            .text("attack_ms")
                            .fixed_decimals(0),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.state.gate_cfg.release_ms, 0.0..=500.0)
                            .text("release_ms")
                            .fixed_decimals(0),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut self.state.pipeline_cfg.sv_update_samples,
                            1_000..=32_000,
                        )
                        .text("sv_update_samples (ECAPA refresh cadence @ 16 kHz)"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut self.state.pipeline_cfg.silence_force_off_ms,
                            0.0..=3000.0,
                        )
                        .text("silence_force_off_ms (0 disables)")
                        .fixed_decimals(0),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.state.pipeline_cfg.score_ema_alpha, 0.0..=1.0)
                            .text("score_ema_alpha (1.0 disables smoothing)")
                            .fixed_decimals(2),
                    );
                    if ui.button("Reset to defaults").clicked() {
                        self.state.gate_cfg = mellonella_core::gating::GateConfig::default();
                        self.state.pipeline_cfg = crate::state::default_live_pipeline_cfg();
                    }
                    ui.separator();
                    ui.label(egui::RichText::new("Enrollment IO").strong());
                    ui.weak(
                        "Mellonella auto-saves the profile to ~/.config/mellonella/enrollment.json. \
                         These buttons exist for CLI interop / advanced flows.",
                    );
                });
                self.render_enrollment_power_user_actions(ui);
                ui.add_enabled_ui(!running, |ui| {
                    ui.separator();
                    ui.label(egui::RichText::new("Stage C — Target Speaker Extraction").strong());
                    ui.horizontal(|ui| {
                        let label = self.state.tse_onnx_path.as_deref().map_or_else(
                            || "(no file selected)".to_string(),
                            |p| {
                                p.file_name().map_or_else(
                                    || p.display().to_string(),
                                    |n| n.to_string_lossy().into_owned(),
                                )
                            },
                        );
                        if ui.button("Pick TSE ONNX…").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("ONNX model", &["onnx"])
                                .pick_file()
                            {
                                self.state.tse_onnx_path = Some(path);
                            }
                        }
                        if ui
                            .button("Download from HuggingFace")
                            .on_hover_text(
                                "Fetch penta2himajin/tse-conv-tasnet-48k from \
                                 huggingface.co into the local cache and use it. \
                                 Subsequent runs reuse the cached file.",
                            )
                            .clicked()
                        {
                            self.state.fetch_tse_from_hf();
                        }
                        ui.label(label);
                        if self.state.tse_onnx_path.is_some() && ui.small_button("Clear").clicked()
                        {
                            self.state.tse_onnx_path = None;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Model variant:");
                        let current = self.state.tse_variant;
                        egui::ComboBox::from_id_salt("tse_variant_combo")
                            .selected_text(current.label())
                            .show_ui(ui, |ui| {
                                for v in [
                                    crate::state::TseVariant::Prod48k,
                                    crate::state::TseVariant::Poc16k,
                                ] {
                                    ui.selectable_value(&mut self.state.tse_variant, v, v.label());
                                }
                            });
                    });
                    ui.label(
                        egui::RichText::new(
                            "Download from huggingface.co/penta2himajin/tse-conv-tasnet-48k",
                        )
                        .small()
                        .color(egui::Color32::GRAY),
                    );
                });
            });
    }

    fn render_error_row(&self, ui: &mut egui::Ui) {
        if let Some(err) = &self.state.last_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
        }
    }
}

impl eframe::App for MellonellaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll the live session for fresh counters / worker errors,
        // and the recorder for completion / progress, before
        // rendering so the UI reflects the latest state.
        self.state.poll_session();
        self.state.poll_recorder();
        self.drain_tray_commands(ctx);
        // Auto-start a session as soon as enrollment + models are
        // ready and the user hasn't manually paused. Idempotent: the
        // call is a no-op once a session is already running.
        self.maybe_auto_start();

        // Step 17: minimise-to-tray. When the user clicks the
        // window's close button AND the tray is available,
        // intercept the close so the live session keeps running in
        // the background. Without a tray (Linux without
        // AppIndicator, etc.) we fall through to the OS default
        // (close = quit) so users aren't trapped in a headless app.
        if ctx.input(|i| i.viewport().close_requested()) && self.tray.is_some() {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            ctx.send_viewport_cmd(egui::ViewportCommand::Visible(false));
            self.window_visible = false;
        }

        // Step 17: keep the tray icon's visual state in sync with
        // the live-session state.
        if let Some(tray) = self.tray.as_mut() {
            tray.set_running(self.state.is_running());
        }

        self.render_central_panel(ctx);

        // Repaint at ~10 Hz so the counter / recording progress
        // displays stay alive even without UI interaction.
        if self.state.is_running() || self.state.is_recording() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}

/// Map RMS in `[0, 1]` to a `[0, 1]` progress-bar reading via a
/// pseudo-log scale: -60 dBFS → 0.0, 0 dBFS → 1.0. Conversational
/// speech sits around -25 dBFS which lights about half the bar —
/// the sweet spot for a level meter that "moves visibly" without
/// pinning at the top.
fn rms_to_bar(rms: f32) -> f32 {
    if rms <= 1e-6 {
        return 0.0;
    }
    let db = 20.0 * rms.log10();
    ((db + 60.0) / 60.0).clamp(0.0, 1.0)
}
