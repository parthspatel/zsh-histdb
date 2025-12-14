{
  lib,
  stdenvNoCC,
  makeWrapper,
  zsh,
  sqlite,
}:

stdenvNoCC.mkDerivation {
  pname = "zsh-histdb";
  version = "0-unstable-2025-12-13";

  src = lib.cleanSource ./.;

  postPatch = ''
    substituteInPlace sqlite-history.zsh \
      --replace-fail 'sqlite3' '"${lib.getExe sqlite}"'
  '';

  buildInputs = [
    zsh
  ];

  installPhase = ''
    runHook preInstall

    install -Dt $out/share/zsh-histdb/ \
      sqlite-history.zsh histdb-interactive.zsh histdb-{merge,migrate}

    runHook postInstall
  '';

  meta = {
    description = "History database for Zsh, based on SQLite";
    homepage = "https://github.com/parthspatel/zsh-histdb-macos";
    license = lib.licenses.mit;
    maintainers = with lib.maintainers; [
      parthspatel
    ];
    platforms = lib.platforms.unix;
  };
}
