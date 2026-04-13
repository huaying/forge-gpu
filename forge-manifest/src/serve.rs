//! Streaming simulation server.
//!
//! `forge run sim.toml --serve 8080` starts a simulation that:
//! 1. Serves a built-in HTML/JS viewer on HTTP (port 8080)
//! 2. Streams particle frames over WebSocket (port 8081)
//!
//! Frame protocol (binary WebSocket messages):
//!   - First 4 bytes: u32 LE = frame number
//!   - Next 4 bytes: u32 LE = particle count
//!   - Remaining: f32 LE × (count × 3) = x,y,z positions

use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

/// Viewer HTML page with Three.js (embedded as const string).
const VIEWER_HTML: &str = include_str!("viewer.html");

/// Start HTTP server to serve the viewer page.
pub fn start_http_server(port: u16) {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).expect("Failed to bind HTTP server");
    println!("  🌐 Viewer: http://localhost:{}", port);

    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            thread::spawn(move || {
                handle_http(stream);
            });
        }
    });
}

fn handle_http(mut stream: TcpStream) {
    use std::io::{Read, Write};
    let mut buf = [0u8; 4096];
    let _ = stream.read(&mut buf);

    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        VIEWER_HTML.len(),
        VIEWER_HTML
    );
    let _ = stream.write_all(response.as_bytes());
}

/// A shared frame buffer that the simulation writes to and WebSocket clients read from.
pub struct FrameBuffer {
    /// Latest frame data: [frame_number, particle_count, x0,y0,z0, x1,y1,z1, ...]
    pub data: Vec<u8>,
    pub frame_number: u32,
    pub ready: bool,
}

impl FrameBuffer {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            frame_number: 0,
            ready: false,
        }
    }

    /// Pack positions into binary frame.
    pub fn set_frame(&mut self, frame_num: u32, positions: &[f32]) {
        let count = (positions.len() / 3) as u32;
        let header_size = 8; // frame_num(4) + count(4)
        let data_size = positions.len() * 4;
        self.data.resize(header_size + data_size, 0);

        // Write header
        self.data[0..4].copy_from_slice(&frame_num.to_le_bytes());
        self.data[4..8].copy_from_slice(&count.to_le_bytes());

        // Write positions
        for (i, &val) in positions.iter().enumerate() {
            let offset = header_size + i * 4;
            self.data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }

        self.frame_number = frame_num;
        self.ready = true;
    }
}

/// Start WebSocket server that streams frames to connected clients.
pub fn start_ws_server(port: u16, frame_buf: Arc<Mutex<FrameBuffer>>) {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).expect("Failed to bind WebSocket server");
    println!("  📡 WebSocket: ws://localhost:{}", port);

    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let buf = Arc::clone(&frame_buf);
            thread::spawn(move || {
                handle_ws_client(stream, buf);
            });
        }
    });
}

fn handle_ws_client(stream: TcpStream, frame_buf: Arc<Mutex<FrameBuffer>>) {
    let mut ws = match tungstenite::accept(stream) {
        Ok(ws) => ws,
        Err(_) => return,
    };

    let mut last_frame = 0u32;

    loop {
        // Check for new frame
        let data = {
            let buf = frame_buf.lock().unwrap();
            if buf.ready && buf.frame_number > last_frame {
                last_frame = buf.frame_number;
                Some(buf.data.clone())
            } else {
                None
            }
        };

        if let Some(data) = data {
            use tungstenite::Message;
            if ws.send(Message::Binary(data.into())).is_err() {
                break; // Client disconnected
            }
        }

        // Small sleep to avoid busy-waiting
        thread::sleep(std::time::Duration::from_millis(8)); // ~120 fps max
    }
}
