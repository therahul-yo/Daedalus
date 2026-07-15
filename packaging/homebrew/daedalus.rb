class Daedalus < Formula
  desc "MacBook-Air-first MLX inference engine: thermally-governed prefill, persistent prefix cache, OpenAI-compatible server"
  homepage "https://github.com/therahul-yo/Daedalus"
  # The canonical tap receives verified binary releases from release.yml.
  # This in-repo template intentionally supports head installs only; claiming
  # a stable artifact without its real SHA256 would be unsafe.
  license "Apache-2.0"
  head "https://github.com/therahul-yo/Daedalus.git", branch: "master"

  depends_on "uv" => :build
  depends_on "mlx"
  depends_on "python@3.12"

  def install
    # Build with PyInstaller
    system "uv", "sync", "--extra", "dev", "--locked"
    system "uv", "pip", "install", "pyinstaller"
    system "uv", "run", "pyinstaller", "--clean", "--noconfirm", "packaging/pyinstaller/daedalus.spec"
    bin.install "dist/daedalus"
  end

  test do
    assert_match "daedalus", shell_output("#{bin}/daedalus --help")
  end
end
