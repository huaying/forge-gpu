//! Streaming simulation server.
//!
//! `forge run sim.toml --serve 8080` starts a simulation that:
//! 1. Serves a built-in HTML/JS viewer on HTTP (port 8080)
//! 2. Streams particle frames over WebSocket (same port 8080, via HTTP Upgrade)
//!
//! Frame protocol (binary WebSocket messages):
//!   - First 4 bytes: u32 LE = frame number
//!   - Next 4 bytes: u32 LE = particle count
//!   - Remaining: f32 LE × (count × 3) = x,y,z positions

use std::io::Read;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

/// Viewer HTML page with Three.js (embedded as const string).
const VIEWER_HTML: &str = include_str!("viewer.html");

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

        self.data[0..4].copy_from_slice(&frame_num.to_le_bytes());
        self.data[4..8].copy_from_slice(&count.to_le_bytes());

        for (i, &val) in positions.iter().enumerate() {
            let offset = header_size + i * 4;
            self.data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
        }

        self.frame_number = frame_num;
        self.ready = true;
    }
}

/// Start a unified server: HTTP for viewer page, WebSocket for frame streaming.
/// Both on the same port — distinguished by HTTP Upgrade header.
pub fn start_server(port: u16, frame_buf: Arc<Mutex<FrameBuffer>>) {
    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).expect("Failed to bind server");
    println!("  🌐 Viewer: http://localhost:{}", port);
    println!("  📡 WebSocket: ws://localhost:{} (same port)", port);

    thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let buf = Arc::clone(&frame_buf);
            thread::spawn(move || {
                handle_connection(stream, buf);
            });
        }
    });
}

fn handle_connection(mut stream: TcpStream, frame_buf: Arc<Mutex<FrameBuffer>>) {
    // Peek at the request to determine if it's a WebSocket upgrade
    let mut peek_buf = [0u8; 4096];
    let n = match stream.read(&mut peek_buf) {
        Ok(n) => n,
        Err(_) => return,
    };
    let request = String::from_utf8_lossy(&peek_buf[..n]);

    if request.contains("Upgrade: websocket") || request.contains("upgrade: websocket") {
        // WebSocket upgrade — use tungstenite with the already-read data
        // We need to re-feed the request to tungstenite
        use std::io::{Cursor, Write};

        // Create a wrapper that first yields the already-read data, then the stream
        let mut ws = match tungstenite::accept(ReadChain::new(&peek_buf[..n], stream)) {
            Ok(ws) => ws,
            Err(_) => return,
        };

        let mut last_frame = 0u32;

        loop {
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
                    break;
                }
            }

            thread::sleep(std::time::Duration::from_millis(8));
        }
    } else {
        // Regular HTTP — serve the viewer page
        use std::io::Write;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store, no-cache, must-revalidate\r\nPragma: no-cache\r\nConnection: close\r\n\r\n{}",
            VIEWER_HTML.len(),
            VIEWER_HTML
        );
        let _ = stream.write_all(response.as_bytes());
    }
}

/// Helper: chain pre-read bytes with the remaining stream.
struct ReadChain<'a> {
    prefix: &'a [u8],
    prefix_pos: usize,
    stream: TcpStream,
}

impl<'a> ReadChain<'a> {
    fn new(prefix: &'a [u8], stream: TcpStream) -> Self {
        Self { prefix, prefix_pos: 0, stream }
    }
}

impl<'a> Read for ReadChain<'a> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.prefix_pos < self.prefix.len() {
            let remaining = &self.prefix[self.prefix_pos..];
            let n = buf.len().min(remaining.len());
            buf[..n].copy_from_slice(&remaining[..n]);
            self.prefix_pos += n;
            Ok(n)
        } else {
            self.stream.read(buf)
        }
    }
}

impl<'a> std::io::Write for ReadChain<'a> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.stream.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.stream.flush()
    }
}
