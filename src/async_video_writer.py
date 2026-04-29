import queue
import threading
import subprocess

class AsyncFFmpegVideoWriter:
    def __init__(self, output_path, codec="h264_nvenc", fps=30, resolution=(640, 480), max_queue_size=0, ffmpeg_path=None, pix_fmt="bgr24", crf=None, preset=None, output_pix_fmt="yuv420p"):
        """
        Multi-threaded FFmpeg Video Writer.

        :param output_path: Path to save video
        :param fps: Frames per second
        :param resolution: (width, height)
        :param codec: Video codec (e.g., "h264_nvenc", "libx264", "av1_nvenc")
        :param max_queue_size: Max number of frames in the queue
        :param ffmpeg_path: Path to FFmpeg executable (default: "ffmpeg" from PATH)
        :param pix_fmt: Pixel format (default: "bgr24" for OpenCV, or "rgb24" for dpg)
        :param crf: Constant Rate Factor for quality (0=lossless, 18=visually lossless, 23=default). Only for libx264.
        :param preset: Encoding preset for libx264 (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        :param output_pix_fmt: Output pixel format (yuv420p=default, yuv444p=no chroma subsampling)
        """
        self.output_path = output_path
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()

        # Use hardcoded FFmpeg path if needed (e.g., Windows issues)
        self.ffmpeg_path = ffmpeg_path

        # Build FFmpeg command based on codec
        cmd = [
            "ffmpeg" if self.ffmpeg_path is None else self.ffmpeg_path,
            "-y",
            "-loglevel", "error",  # Only show errors, suppress progress/info
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{resolution[0]}x{resolution[1]}",  # Resolution
            "-pix_fmt", pix_fmt,
            "-r", str(fps),  # FPS
            "-i", "-",  # Read from stdin
            "-c:v", codec,  # Use GPU or CPU encoder
        ]

        # Add codec-specific options
        if "nvenc" in codec:
            # NVENC-specific options
            cmd.extend([
                "-preset", "p7",  # Fastest NVENC preset
                "-cq", "16",
                "-qp", "16",
                "-rc", "vbr",
            ])
        elif codec == "libx264":
            # libx264-specific options
            cmd.extend([
                "-preset", preset if preset is not None else "medium",  # Encoding speed vs compression
                "-crf", str(crf if crf is not None else 18),  # 0=lossless, 18=visually lossless (default), 23=standard
            ])

        # Output pixel format and path
        cmd.extend([
            "-pix_fmt", output_pix_fmt,  # Output format (yuv420p or yuv444p)
            output_path,
        ])

        # Start FFmpeg process
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr for error messages only
        )

        # Start worker thread
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def _writer_worker(self):
        """Worker thread for writing frames to FFmpeg."""
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(block=True, timeout=0.1)
            except queue.Empty:
                continue

            try:
                self.process.stdin.write(frame.tobytes())  # Write frame data
                self.process.stdin.flush()  # Ensure immediate writing
            except (BrokenPipeError, OSError) as e:
                # FFmpeg process died, stop processing
                print(f"[ThreadedFFmpegVideoWriter] Error writing frame: {e}")
                # Drain the queue to unblock any waiting writers
                while not self.queue.empty():
                    try:
                        self.queue.get_nowait()
                        self.queue.task_done()
                    except queue.Empty:
                        break
                break
            self.queue.task_done()

    def write(self, frame):
        """
        Adds a frame to the queue (blocks if full).

        :param frame: Frame (numpy array) to write to video
        """
        if self.queue.full():
            print("[ThreadedFFmpegVideoWriter] Queue full! Blocking until space is available...")

        self.queue.put(frame)  # Blocking write

    def stop(self):
        """Stops writing and ensures all frames are saved."""
        self.stop_event.set()
        self.writer_thread.join()

        # Close stdin to signal EOF to FFmpeg
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # Process already closed

        # Wait for FFmpeg to finish - it should exit cleanly after receiving EOF
        try:
            returncode = self.process.wait(timeout=10.0)  # Should finish quickly with minimal stderr
        except subprocess.TimeoutExpired:
            print("[ThreadedFFmpegVideoWriter] WARNING: FFmpeg did not finish in 10s, terminating...")
            self.process.terminate()
            try:
                returncode = self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                print("[ThreadedFFmpegVideoWriter] FFmpeg still running, killing...")
                self.process.kill()
                returncode = self.process.wait()

        # Only print errors if there was an actual failure
        if returncode != 0:
            try:
                stderr_output = self.process.stderr.read().decode('utf-8', errors='ignore')
                if stderr_output:  # Only print if there's actual error output
                    print(f"[ThreadedFFmpegVideoWriter] FFmpeg exited with code {returncode}")
                    print(f"[ThreadedFFmpegVideoWriter] FFmpeg stderr:\n{stderr_output}")
            except:
                pass
