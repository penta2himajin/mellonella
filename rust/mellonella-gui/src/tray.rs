//! System-tray integration. Builds a `tray_icon::TrayIcon` with a
//! Show / Toggle / Quit menu and forwards user clicks to the eframe
//! app via a non-blocking channel.
//!
//! Step 17 polish: two icon variants (idle / running) so users can
//! tell at a glance whether the live filter is active without
//! opening the window. The app calls [`TrayHandles::set_running`]
//! whenever the session state changes.
//!
//! Platforms vary in how well the system tray is supported. On
//! macOS / Windows the icon shows up in the global tray. On Linux,
//! `tray-icon` uses the AppIndicator / DBus protocol — works on KDE,
//! Cinnamon, MATE etc. out of the box and on GNOME with the
//! AppIndicator extension. The crate must still build everywhere
//! (CI runs on headless Linux), so failure to construct the tray is
//! a soft error: we log and continue with just the window.

use std::sync::mpsc::{channel, Receiver, Sender};

use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};

/// Commands the tray menu emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayCommand {
    /// Bring the main window into view and focus it.
    Show,
    /// Start the live session if stopped; stop if running. Same as
    /// clicking the main window's Start/Stop button.
    Toggle,
    /// Quit the app.
    Quit,
}

/// Bundle of tray-icon handles + a receiver that yields commands as
/// the user clicks menu items. Keep this alive for the whole app
/// lifetime; dropping it removes the tray icon.
pub struct TrayHandles {
    tray: TrayIcon,
    rx: Receiver<TrayCommand>,
    // IDs let the menu-event poller route clicks to the right
    // TrayCommand. Kept here to avoid lazy_static / global maps.
    show_id: String,
    toggle_id: String,
    quit_id: String,
    forwarder_tx: Sender<TrayCommand>,
    /// Cached icon variants — building these once on init avoids
    /// re-allocating + re-encoding the 32×32 RGBA buffer every
    /// time the session toggles.
    idle_icon: Icon,
    running_icon: Icon,
    /// Last `set_running` value, to skip redundant icon swaps.
    is_running: bool,
}

impl TrayHandles {
    /// Try to build a tray icon and start the menu-event forwarder.
    /// Returns `None` on platforms / sessions where tray-icon
    /// initialisation fails — the caller falls back to a
    /// window-only experience.
    pub fn try_new() -> Option<Self> {
        let idle_icon = build_icon(IconKind::Idle).ok()?;
        let running_icon = build_icon(IconKind::Running).ok()?;
        // Clone here so we can keep both originals for later
        // re-application — `tray-icon` consumes the Icon on each
        // `set_icon` call.
        let idle_icon_init = build_icon(IconKind::Idle).ok()?;

        let menu = Menu::new();
        let show_item = MenuItem::new("Show window", true, None);
        let toggle_item = MenuItem::new("Start / Stop", true, None);
        let quit_item = MenuItem::new("Quit", true, None);
        menu.append(&show_item).ok()?;
        menu.append(&toggle_item).ok()?;
        menu.append(&quit_item).ok()?;

        let tray = TrayIconBuilder::new()
            .with_tooltip("Mellonella — idle")
            .with_menu(Box::new(menu))
            .with_icon(idle_icon_init)
            .build()
            .ok()?;

        let (tx, rx) = channel();
        let show_id = show_item.id().0.clone();
        let toggle_id = toggle_item.id().0.clone();
        let quit_id = quit_item.id().0.clone();
        Some(Self {
            tray,
            rx,
            show_id,
            toggle_id,
            quit_id,
            forwarder_tx: tx,
            idle_icon,
            running_icon,
            is_running: false,
        })
    }

    /// Switch the tray icon + tooltip to reflect the live-session
    /// state. No-op when the state hasn't changed.
    pub fn set_running(&mut self, running: bool) {
        if self.is_running == running {
            return;
        }
        self.is_running = running;
        let (icon, tooltip) = if running {
            (&self.running_icon, "Mellonella — running")
        } else {
            (&self.idle_icon, "Mellonella — idle")
        };
        let _ = self.tray.set_icon(Some(icon.clone()));
        let _ = self.tray.set_tooltip(Some(tooltip));
    }

    /// Drain any pending tray-menu events into the local channel,
    /// then return one command if available. Designed to be called
    /// once per egui frame (10–60 Hz) so it never blocks.
    pub fn try_recv(&self) -> Option<TrayCommand> {
        // `tray-icon` publishes menu events to a global static
        // receiver. Pump it into our owned channel each call so the
        // mapping from event id → TrayCommand stays local to this
        // module.
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            let id = ev.id.0.as_str();
            let cmd = if id == self.show_id {
                Some(TrayCommand::Show)
            } else if id == self.toggle_id {
                Some(TrayCommand::Toggle)
            } else if id == self.quit_id {
                Some(TrayCommand::Quit)
            } else {
                None
            };
            if let Some(c) = cmd {
                let _ = self.forwarder_tx.send(c);
            }
        }
        self.rx.try_recv().ok()
    }
}

/// Tray-icon visual variant: the live-session is idle (filter off)
/// or running (filter on). Procedurally drawn so we don't ship two
/// PNG assets for one binary.
#[derive(Debug, Clone, Copy)]
enum IconKind {
    Idle,
    Running,
}

/// Build a 32×32 RGBA icon procedurally. Idle is the wax-moth-
/// yellow disc that's shipped since step 16; Running adds a
/// green outer ring on top of the yellow disc so the tray-area
/// "is the filter on?" status is glanceable.
fn build_icon(kind: IconKind) -> Result<Icon, tray_icon::BadIcon> {
    const SIZE: u32 = 32;
    let mut rgba = vec![0_u8; (SIZE * SIZE * 4) as usize];
    let cx = SIZE as f32 / 2.0 - 0.5;
    let cy = SIZE as f32 / 2.0 - 0.5;
    let outer_r = SIZE as f32 / 2.0 - 1.0;
    // Inner disc shrinks slightly for the running variant so the
    // outer green ring is visible.
    let inner_r = match kind {
        IconKind::Idle => outer_r,
        IconKind::Running => outer_r - 4.0,
    };

    for y in 0..SIZE {
        for x in 0..SIZE {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            let idx = ((y * SIZE + x) * 4) as usize;
            if d <= inner_r {
                // mellonella → wax-moth yellow.
                rgba[idx] = 230;
                rgba[idx + 1] = 200;
                rgba[idx + 2] = 80;
                rgba[idx + 3] = 255;
            } else if matches!(kind, IconKind::Running) && d <= outer_r {
                // Green ring for "live session active".
                rgba[idx] = 80;
                rgba[idx + 1] = 200;
                rgba[idx + 2] = 120;
                rgba[idx + 3] = 255;
            }
        }
    }
    Icon::from_rgba(rgba, SIZE, SIZE)
}
