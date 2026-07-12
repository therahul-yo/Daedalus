class Daedalus < Formula
  desc "MacBook-Air-first MLX inference engine: thermally-governed prefill, persistent prefix cache, OpenAI-compatible server"
  homepage "https://github.com/nousresearch/daedalus"
  url "https://github.com/nousresearch/daedalus/releases/download/v0.0.1/daedalus-0.0.1-macos-arm64.tar.gz"
  sha256 "REPLACE_WITH_SHA256_OF_RELEASE_TARBALL"
  license "Apache-2.0"
  version "0.0.1"

  depends_on "python@3.12"
  depends_on "mlx"
  depends_on "mlx-lm"

  def install
    libexec.install Dir["*"]
    bin.install_symlink libexec/"daedalus"
  end

  test do
    system "#{bin}/daedalus", "doctor"
  end
end