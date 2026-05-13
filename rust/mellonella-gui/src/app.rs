//! `eframe::App` implementation — the egui UI rendering loop.
//!
//! Single-window layout, minimal styling. Tray-side menu events
//! (Show / Start / Stop / Quit) come in via the channel the
//! `tray` module installs; we drain it on each `update()` call.

use eframe::egui;

use crate::state::{AppState, OUTPUT_SAMPLE_RATE};
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

            self.render_enrollment_row(ui);
            ui.add_space(6.0);
            self.render_device_row(ui);
            ui.add_space(10.0);
            ui.separator();
            ui.add_space(8.0);
            self.render_run_controls(ui);
            ui.add_space(8.0);
            self.render_status(ui);
            ui.add_space(6.0);
            self.render_error_row(ui);
        });
    }

    fn render_enrollment_row(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Enrollment:");
            let label = self
                .state
                .enrollment_path
                .as_ref()
                .map_or("(none)".to_string(), |p| {
                    p.file_name().map_or_else(
                        || p.display().to_string(),
                        |n| n.to_string_lossy().into_owned(),
                    )
                });
            ui.monospace(label);
            if ui.button("Browse…").clicked() {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("Enrollment JSON", &["json"])
                    .pick_file()
                {
                    self.state.load_enrollment(&path);
                }
            }
        });
        if self.state.pool_anchors > 0 {
            ui.label(format!(
                "  {} anchors · F0 μ={:.1} Hz · σ={:.1} Hz",
                self.state.pool_anchors, self.state.pool_f0_mu, self.state.pool_f0_sigma,
            ));
        }
    }

    fn render_device_row(&mut self, ui: &mut egui::Ui) {
        let running = self.state.is_running();
        ui.horizontal(|ui| {
            ui.label("Input:");
            let current = self
                .state
                .selected_input
                .clone()
                .unwrap_or_else(|| "(default)".into());
            ui.add_enabled_ui(!running, |ui| {
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
            ui.add_enabled_ui(!running, |ui| {
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
                .add_enabled(!running, egui::Button::new("Refresh"))
                .clicked()
            {
                self.state.refresh_devices();
            }
        });
    }

    fn render_run_controls(&mut self, ui: &mut egui::Ui) {
        let running = self.state.is_running();
        let can_start = self.state.can_start();
        ui.horizontal(|ui| {
            let label = if running { "Stop" } else { "Start" };
            let enabled = if running { true } else { can_start };
            if ui
                .add_enabled(
                    enabled,
                    egui::Button::new(label).min_size(egui::vec2(120.0, 32.0)),
                )
                .clicked()
            {
                if running {
                    self.state.stop();
                } else {
                    self.state.start();
                }
            }
            let status = if running {
                egui::RichText::new("● running").color(egui::Color32::from_rgb(80, 200, 120))
            } else if can_start {
                egui::RichText::new("○ ready").color(egui::Color32::from_rgb(220, 200, 80))
            } else {
                egui::RichText::new("○ idle").color(egui::Color32::DARK_GRAY)
            };
            ui.label(status);
        });
    }

    fn render_status(&self, ui: &mut egui::Ui) {
        let s = self.state.last_stats;
        let secs = s.samples_processed as f32 / OUTPUT_SAMPLE_RATE as f32;
        ui.label(format!(
            "Processed: {secs:.1} s   ·   overruns: {}   underruns: {}",
            s.input_overruns, s.output_underruns
        ));
    }

    fn render_error_row(&self, ui: &mut egui::Ui) {
        if let Some(err) = &self.state.last_error {
            ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
        }
    }
}

impl eframe::App for MellonellaApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll the live session for fresh counters / worker errors
        // before rendering so the UI reflects the latest state in
        // this frame.
        self.state.poll_session();
        self.drain_tray_commands(ctx);
        self.render_central_panel(ctx);

        // Repaint at ~10 Hz so the counter display stays alive even
        // without UI interaction.
        if self.state.is_running() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
}
