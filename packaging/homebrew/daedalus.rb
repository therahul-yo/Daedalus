class Daedalus < Formula
  desc "MacBook-Air-first MLX inference engine: thermally-governed prefill, persistent prefix cache, OpenAI-compatible server"
  homepage "https://github.com/therahul-yo/daedalus"
  url "https://github.com/therahul-yo/daedalus/archive/refs/tags/v0.2.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "Apache-2.0"
  head "https://github.com/therahul-yo/daedalus.git", branch: "master"

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