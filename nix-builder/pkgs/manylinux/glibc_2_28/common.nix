/*
  Build configuration used to build glibc, Info files, and locale
   information.

   Note that this derivation has multiple outputs and does not respect the
   standard convention of putting the executables into the first output. The
   first output is `lib` so that the libraries provided by this derivation
   can be accessed directly, e.g.

     "${pkgs.glibc}/lib/ld-linux-x86_64.so.2"

   The executables are put into `bin` output and need to be referenced via
   the `bin` attribute of the main package, e.g.

     "${pkgs.glibc.bin}/bin/ldd".

  The executables provided by glibc typically include `ldd`, `locale`, `iconv`
  but the exact set depends on the library version and the configuration.
*/

{
  stdenv,
  lib,
  buildPackages,
  fetchurl,
  fetchpatch,
  linuxHeaders ? null,
  gd ? null,
  libpng ? null,
  bison,
}:

{
  name,
  withLinuxHeaders ? false,
  profilingLibraries ? false,
  withGd ? false,
  meta,
  ...
}@args:

let
  version = "2.28";
  patchSuffix = "";
  hash = "sha256-sZAAUa+tdvek9z5xQT30gm3OCF743beFqUW2bX1RMII=";
in

assert withLinuxHeaders -> linuxHeaders != null;
assert withGd -> gd != null && libpng != null;

stdenv.mkDerivation (
  {
    inherit version;
    linuxHeaders = if withLinuxHeaders then linuxHeaders else null;

    inherit (stdenv) is64bit;

    enableParallelBuilding = true;

    patches = [
      # Have rpcgen(1) look for cpp(1) in $PATH.
      ./rpcgen-path.patch

      # Allow NixOS and Nix to handle the locale-archive.
      ./nix-locale-archive.patch

      # Don't use /etc/ld.so.preload, but /etc/ld-nix.so.preload.
      ./dont-use-system-ld-so-preload.patch

      /*
        The command "getconf CS_PATH" returns the default search path
        "/bin:/usr/bin", which is inappropriate on NixOS machines. This
        patch extends the search path by "/run/current-system/sw/bin".
      */
      ./fix_path_attribute_in_getconf.patch

      /*
        Allow running with RHEL 6 -like kernels.  The patch adds an exception
        for glibc to accept 2.6.32 and to tag the ELFs as 2.6.32-compatible
        (otherwise the loader would refuse libc).
        Note that glibc will fully work only on their heavily patched kernels
        and we lose early mismatch detection on 2.6.32.

        On major glibc updates we should check that the patched kernel supports
        all the required features.  ATM it's verified up to glibc-2.26-131.
        # HOWTO: check glibc sources for changes in kernel requirements
        git log -p glibc-2.25.. sysdeps/unix/sysv/linux/x86_64/kernel-features.h sysdeps/unix/sysv/linux/kernel-features.h
        # get kernel sources (update the URL)
        mkdir tmp && cd tmp
        curl http://vault.centos.org/6.9/os/Source/SPackages/kernel-2.6.32-696.el6.src.rpm | rpm2cpio - | cpio -idmv
        tar xf linux-*.bz2
        # check syscall presence, for example
        less linux-*?/arch/x86/kernel/syscall_table_32.S
      */
      ./allow-kernel-2.6.32.patch
      # Provide utf-8 locales by default, so we can use it in stdenv without depending on our large locale-archive.
      (fetchurl {
        url = "https://salsa.debian.org/glibc-team/glibc/raw/49767c9f7de4828220b691b29de0baf60d8a54ec/debian/patches/localedata/locale-C.diff";
        sha256 = "0irj60hs2i91ilwg5w7sqrxb695c93xg0ik7yhhq9irprd7fidn4";
      })
    ]
    ++ lib.optionals stdenv.isx86_64 [
      ./fix-x64-abi.patch
    ]
    ++ lib.optional stdenv.hostPlatform.isMusl ./fix-rpc-types-musl-conflicts.patch
    ++ lib.optional stdenv.buildPlatform.isDarwin ./darwin-cross-build.patch

    # Remove after upgrading to glibc 2.28+
    ++
      lib.optional (stdenv.hostPlatform != stdenv.buildPlatform || stdenv.hostPlatform.isMusl)
        (fetchpatch {
          url = "https://sourceware.org/git/?p=glibc.git;a=patch;h=780684eb04298977bc411ebca1eadeeba4877833";
          name = "correct-pwent-parsing-issue-and-resulting-build.patch";
          sha256 = "08fja894vzaj8phwfhsfik6jj2pbji7kypy3q8pgxvsd508zdv1q";
          excludes = [ "ChangeLog" ];
        })
    ++ [
      ./rhel/glibc-fedora-nscd.patch
      ./rhel/glibc-rh697421.patch
      ./rhel/glibc-fedora-linux-tcsetattr.patch
      ./rhel/glibc-rh741105.patch
      ./rhel/glibc-fedora-localedef.patch
      ./rhel/glibc-fedora-nis-rh188246.patch
      ./rhel/glibc-fedora-manual-dircategory.patch
      ./rhel/glibc-rh827510.patch
      ./rhel/glibc-fedora-locarchive.patch
      ./rhel/glibc-fedora-streams-rh436349.patch
      ./rhel/glibc-rh819430.patch
      ./rhel/glibc-fedora-localedata-rh61908.patch
      ./rhel/glibc-fedora-__libc_multiple_libcs.patch
      ./rhel/glibc-rh1070416.patch
      ./rhel/glibc-nscd-sysconfig.patch
      # Keep this one disabled, conflicts with nixpkgs path patch.
      # ./rhel/glibc-cs-path.patch
      # Conflicts with nixpkgs locale patch.
      # ./rhel/glibc-c-utf8-locale.patch
      ./rhel/glibc-python3.patch
      ./rhel/glibc-with-nonshared-cflags.patch
      ./rhel/glibc-asflags.patch
      ./rhel/glibc-rh1614253.patch
      ./rhel/glibc-rh1577365.patch
      ./rhel/glibc-rh1615781.patch
      ./rhel/glibc-rh1615784.patch
      ./rhel/glibc-rh1615790.patch
      ./rhel/glibc-rh1622675.patch
      ./rhel/glibc-rh1622678-1.patch
      ./rhel/glibc-rh1622678-2.patch
      ./rhel/glibc-rh1631293-1.patch
      ./rhel/glibc-rh1631293-2.patch
      ./rhel/glibc-rh1623536.patch
      ./rhel/glibc-rh1631722.patch
      ./rhel/glibc-rh1631730.patch
      ./rhel/glibc-rh1623536-2.patch
      ./rhel/glibc-rh1614979.patch
      ./rhel/glibc-rh1645593.patch
      ./rhel/glibc-rh1645596.patch
      ./rhel/glibc-rh1645604.patch
      ./rhel/glibc-rh1646379.patch
      ./rhel/glibc-rh1645601.patch
      ./rhel/glibc-rh1638523-1.patch
      ./rhel/glibc-rh1638523-2.patch
      ./rhel/glibc-rh1638523-3.patch
      ./rhel/glibc-rh1638523-4.patch
      ./rhel/glibc-rh1638523-5.patch
      ./rhel/glibc-rh1638523-6.patch
      ./rhel/glibc-rh1641982.patch
      ./rhel/glibc-rh1645597.patch
      ./rhel/glibc-rh1650560-1.patch
      ./rhel/glibc-rh1650560-2.patch
      ./rhel/glibc-rh1650563.patch
      ./rhel/glibc-rh1650566.patch
      ./rhel/glibc-rh1650571.patch
      ./rhel/glibc-rh1638520.patch
      ./rhel/glibc-rh1651274.patch
      ./rhel/glibc-rh1654010-1.patch
      ./rhel/glibc-rh1635779.patch
      ./rhel/glibc-rh1654010-2.patch
      ./rhel/glibc-rh1654010-3.patch
      ./rhel/glibc-rh1654010-4.patch
      ./rhel/glibc-rh1654010-5.patch
      ./rhel/glibc-rh1654010-6.patch
      ./rhel/glibc-rh1642094-1.patch
      ./rhel/glibc-rh1642094-2.patch
      ./rhel/glibc-rh1642094-3.patch
      ./rhel/glibc-rh1654872-1.patch
      ./rhel/glibc-rh1654872-2.patch
      ./rhel/glibc-rh1651283-1.patch
      ./rhel/glibc-rh1662843-1.patch
      ./rhel/glibc-rh1662843-2.patch
      ./rhel/glibc-rh1623537.patch
      ./rhel/glibc-rh1577438.patch
      ./rhel/glibc-rh1664408.patch
      ./rhel/glibc-rh1651742.patch
      ./rhel/glibc-rh1672773.patch
      ./rhel/glibc-rh1651283-2.patch
      ./rhel/glibc-rh1651283-3.patch
      ./rhel/glibc-rh1651283-4.patch
      ./rhel/glibc-rh1651283-5.patch
      ./rhel/glibc-rh1651283-6.patch
      ./rhel/glibc-rh1651283-7.patch
      ./rhel/glibc-rh1659293-1.patch
      ./rhel/glibc-rh1659293-2.patch
      ./rhel/glibc-rh1639343-1.patch
      ./rhel/glibc-rh1639343-2.patch
      ./rhel/glibc-rh1639343-3.patch
      ./rhel/glibc-rh1639343-4.patch
      ./rhel/glibc-rh1639343-5.patch
      ./rhel/glibc-rh1639343-6.patch
      ./rhel/glibc-rh1663035.patch
      ./rhel/glibc-rh1658901.patch
      ./rhel/glibc-rh1659512-1.patch
      ./rhel/glibc-rh1659512-2.patch
      ./rhel/glibc-rh1659438-1.patch
      ./rhel/glibc-rh1659438-2.patch
      ./rhel/glibc-rh1659438-3.patch
      ./rhel/glibc-rh1659438-4.patch
      ./rhel/glibc-rh1659438-5.patch
      ./rhel/glibc-rh1659438-6.patch
      ./rhel/glibc-rh1659438-7.patch
      ./rhel/glibc-rh1659438-8.patch
      ./rhel/glibc-rh1659438-9.patch
      ./rhel/glibc-rh1659438-10.patch
      ./rhel/glibc-rh1659438-11.patch
      ./rhel/glibc-rh1659438-12.patch
      ./rhel/glibc-rh1659438-13.patch
      ./rhel/glibc-rh1659438-14.patch
      ./rhel/glibc-rh1659438-15.patch
      ./rhel/glibc-rh1659438-16.patch
      ./rhel/glibc-rh1659438-17.patch
      ./rhel/glibc-rh1659438-18.patch
      ./rhel/glibc-rh1659438-19.patch
      ./rhel/glibc-rh1659438-20.patch
      ./rhel/glibc-rh1659438-21.patch
      ./rhel/glibc-rh1659438-22.patch
      ./rhel/glibc-rh1659438-23.patch
      ./rhel/glibc-rh1659438-24.patch
      ./rhel/glibc-rh1659438-25.patch
      ./rhel/glibc-rh1659438-26.patch
      ./rhel/glibc-rh1659438-27.patch
      ./rhel/glibc-rh1659438-28.patch
      ./rhel/glibc-rh1659438-29.patch
      ./rhel/glibc-rh1659438-30.patch
      ./rhel/glibc-rh1659438-31.patch
      ./rhel/glibc-rh1659438-32.patch
      ./rhel/glibc-rh1659438-33.patch
      ./rhel/glibc-rh1659438-34.patch
      ./rhel/glibc-rh1659438-35.patch
      ./rhel/glibc-rh1659438-36.patch
      ./rhel/glibc-rh1659438-37.patch
      ./rhel/glibc-rh1659438-38.patch
      ./rhel/glibc-rh1659438-39.patch
      ./rhel/glibc-rh1659438-40.patch
      ./rhel/glibc-rh1659438-41.patch
      ./rhel/glibc-rh1659438-42.patch
      ./rhel/glibc-rh1659438-43.patch
      ./rhel/glibc-rh1659438-44.patch
      ./rhel/glibc-rh1659438-45.patch
      ./rhel/glibc-rh1659438-46.patch
      ./rhel/glibc-rh1659438-47.patch
      ./rhel/glibc-rh1659438-48.patch
      ./rhel/glibc-rh1659438-49.patch
      ./rhel/glibc-rh1659438-50.patch
      ./rhel/glibc-rh1659438-51.patch
      ./rhel/glibc-rh1659438-52.patch
      ./rhel/glibc-rh1659438-53.patch
      ./rhel/glibc-rh1659438-54.patch
      ./rhel/glibc-rh1659438-55.patch
      ./rhel/glibc-rh1659438-56.patch
      ./rhel/glibc-rh1659438-57.patch
      ./rhel/glibc-rh1659438-58.patch
      ./rhel/glibc-rh1659438-59.patch
      ./rhel/glibc-rh1659438-60.patch
      ./rhel/glibc-rh1659438-61.patch
      ./rhel/glibc-rh1659438-62.patch
      ./rhel/glibc-rh1702539-1.patch
      ./rhel/glibc-rh1702539-2.patch
      ./rhel/glibc-rh1701605-1.patch
      ./rhel/glibc-rh1701605-2.patch
      ./rhel/glibc-rh1691528-1.patch
      ./rhel/glibc-rh1691528-2.patch
      ./rhel/glibc-rh1706777.patch
      ./rhel/glibc-rh1710478.patch
      ./rhel/glibc-rh1670043-1.patch
      ./rhel/glibc-rh1670043-2.patch
      ./rhel/glibc-rh1710894.patch
      ./rhel/glibc-rh1699194-1.patch
      ./rhel/glibc-rh1699194-2.patch
      ./rhel/glibc-rh1699194-3.patch
      ./rhel/glibc-rh1699194-4.patch
      ./rhel/glibc-rh1727241-1.patch
      ./rhel/glibc-rh1727241-2.patch
      ./rhel/glibc-rh1727241-3.patch
      ./rhel/glibc-rh1717438.patch
      ./rhel/glibc-rh1727152.patch
      ./rhel/glibc-rh1724975.patch
      ./rhel/glibc-rh1722215.patch
      ./rhel/glibc-rh1764234-1.patch
      ./rhel/glibc-rh1764234-2.patch
      ./rhel/glibc-rh1764234-3.patch
      ./rhel/glibc-rh1764234-4.patch
      ./rhel/glibc-rh1764234-5.patch
      ./rhel/glibc-rh1764234-6.patch
      ./rhel/glibc-rh1764234-7.patch
      ./rhel/glibc-rh1764234-8.patch
      ./rhel/glibc-rh1747505-1.patch
      ./rhel/glibc-rh1747505-2.patch
      ./rhel/glibc-rh1747505-3.patch
      ./rhel/glibc-rh1747505-4.patch
      ./rhel/glibc-rh1747453.patch
      ./rhel/glibc-rh1764241.patch
      ./rhel/glibc-rh1746933-1.patch
      ./rhel/glibc-rh1746933-2.patch
      ./rhel/glibc-rh1746933-3.patch
      ./rhel/glibc-rh1735747-1.patch
      ./rhel/glibc-rh1735747-2.patch
      ./rhel/glibc-rh1764226-1.patch
      ./rhel/glibc-rh1764226-2.patch
      ./rhel/glibc-rh1764226-3.patch
      ./rhel/glibc-rh1764218-1.patch
      ./rhel/glibc-rh1764218-2.patch
      ./rhel/glibc-rh1764218-3.patch
      ./rhel/glibc-rh1682954.patch
      ./rhel/glibc-rh1746928.patch
      ./rhel/glibc-rh1747502.patch
      ./rhel/glibc-rh1747502-1.patch
      ./rhel/glibc-rh1747502-2.patch
      ./rhel/glibc-rh1747502-3.patch
      ./rhel/glibc-rh1747502-4.patch
      ./rhel/glibc-rh1747502-5.patch
      ./rhel/glibc-rh1747502-6.patch
      ./rhel/glibc-rh1747502-7.patch
      ./rhel/glibc-rh1747502-8.patch
      ./rhel/glibc-rh1747502-9.patch
      ./rhel/glibc-rh1726638-1.patch
      ./rhel/glibc-rh1726638-2.patch
      ./rhel/glibc-rh1726638-3.patch
      ./rhel/glibc-rh1764238-1.patch
      ./rhel/glibc-rh1764238-2.patch
      ./rhel/glibc-rh1764242.patch
      ./rhel/glibc-rh1769304.patch
      ./rhel/glibc-rh1749439-1.patch
      ./rhel/glibc-rh1749439-2.patch
      ./rhel/glibc-rh1749439-3.patch
      ./rhel/glibc-rh1749439-4.patch
      ./rhel/glibc-rh1749439-5.patch
      ./rhel/glibc-rh1749439-6.patch
      ./rhel/glibc-rh1749439-7.patch
      ./rhel/glibc-rh1749439-8.patch
      ./rhel/glibc-rh1749439-9.patch
      ./rhel/glibc-rh1749439-10.patch
      ./rhel/glibc-rh1749439-11.patch
      ./rhel/glibc-rh1749439-12.patch
      ./rhel/glibc-rh1749439-13.patch
      ./rhel/glibc-rh1764231-1.patch
      ./rhel/glibc-rh1764231-2.patch
      ./rhel/glibc-rh1764235.patch
      # Patch for disabled locale patch.
      #./rhel/glibc-rh1361965.patch
      ./rhel/glibc-rh1764223.patch
      ./rhel/glibc-rh1764214.patch
      ./rhel/glibc-rh1774021.patch
      ./rhel/glibc-rh1775294.patch
      ./rhel/glibc-rh1777241.patch
      ./rhel/glibc-rh1410154-1.patch
      ./rhel/glibc-rh1410154-2.patch
      ./rhel/glibc-rh1410154-3.patch
      ./rhel/glibc-rh1410154-4.patch
      ./rhel/glibc-rh1410154-5.patch
      ./rhel/glibc-rh1410154-6.patch
      ./rhel/glibc-rh1410154-7.patch
      ./rhel/glibc-rh1410154-8.patch
      ./rhel/glibc-rh1410154-9.patch
      ./rhel/glibc-rh1410154-10.patch
      ./rhel/glibc-rh1410154-11.patch
      ./rhel/glibc-rh1410154-12.patch
      ./rhel/glibc-rh1410154-13.patch
      ./rhel/glibc-rh1410154-14.patch
      ./rhel/glibc-rh1410154-15.patch
      ./rhel/glibc-rh1410154-16.patch
      ./rhel/glibc-rh1810142-1.patch
      ./rhel/glibc-rh1810142-2.patch
      ./rhel/glibc-rh1810142-3.patch
      ./rhel/glibc-rh1810142-4.patch
      ./rhel/glibc-rh1810142-5.patch
      ./rhel/glibc-rh1810142-6.patch
      ./rhel/glibc-rh1743445-1.patch
      ./rhel/glibc-rh1743445-2.patch
      ./rhel/glibc-rh1780204-01.patch
      ./rhel/glibc-rh1780204-02.patch
      ./rhel/glibc-rh1780204-03.patch
      ./rhel/glibc-rh1780204-04.patch
      ./rhel/glibc-rh1780204-05.patch
      ./rhel/glibc-rh1780204-06.patch
      ./rhel/glibc-rh1780204-07.patch
      ./rhel/glibc-rh1780204-08.patch
      ./rhel/glibc-rh1780204-09.patch
      ./rhel/glibc-rh1780204-10.patch
      ./rhel/glibc-rh1780204-11.patch
      ./rhel/glibc-rh1780204-12.patch
      ./rhel/glibc-rh1780204-13.patch
      ./rhel/glibc-rh1780204-14.patch
      ./rhel/glibc-rh1780204-15.patch
      ./rhel/glibc-rh1780204-16.patch
      ./rhel/glibc-rh1780204-17.patch
      ./rhel/glibc-rh1780204-18.patch
      ./rhel/glibc-rh1780204-19.patch
      ./rhel/glibc-rh1780204-20.patch
      ./rhel/glibc-rh1780204-21.patch
      ./rhel/glibc-rh1780204-22.patch
      ./rhel/glibc-rh1780204-23.patch
      ./rhel/glibc-rh1780204-24.patch
      ./rhel/glibc-rh1780204-25.patch
      ./rhel/glibc-rh1780204-26.patch
      ./rhel/glibc-rh1780204-27.patch
      ./rhel/glibc-rh1780204-28.patch
      ./rhel/glibc-rh1784519.patch
      ./rhel/glibc-rh1775819.patch
      ./rhel/glibc-rh1774114.patch
      ./rhel/glibc-rh1812756-1.patch
      ./rhel/glibc-rh1812756-2.patch
      ./rhel/glibc-rh1812756-3.patch
      ./rhel/glibc-rh1757354.patch
      ./rhel/glibc-rh1784520.patch
      ./rhel/glibc-rh1784525.patch
      ./rhel/glibc-rh1810146.patch
      ./rhel/glibc-rh1810223-1.patch
      ./rhel/glibc-rh1810223-2.patch
      ./rhel/glibc-rh1811796-1.patch
      ./rhel/glibc-rh1811796-2.patch
      ./rhel/glibc-rh1813398.patch
      ./rhel/glibc-rh1813399.patch
      ./rhel/glibc-rh1810224-1.patch
      ./rhel/glibc-rh1810224-2.patch
      ./rhel/glibc-rh1810224-3.patch
      ./rhel/glibc-rh1810224-4.patch
      ./rhel/glibc-rh1783303-1.patch
      ./rhel/glibc-rh1783303-2.patch
      ./rhel/glibc-rh1783303-3.patch
      ./rhel/glibc-rh1783303-4.patch
      ./rhel/glibc-rh1783303-5.patch
      ./rhel/glibc-rh1783303-6.patch
      ./rhel/glibc-rh1783303-7.patch
      ./rhel/glibc-rh1783303-8.patch
      ./rhel/glibc-rh1783303-9.patch
      ./rhel/glibc-rh1783303-10.patch
      ./rhel/glibc-rh1783303-11.patch
      ./rhel/glibc-rh1783303-12.patch
      ./rhel/glibc-rh1783303-13.patch
      ./rhel/glibc-rh1783303-14.patch
      ./rhel/glibc-rh1783303-15.patch
      ./rhel/glibc-rh1783303-16.patch
      ./rhel/glibc-rh1783303-17.patch
      ./rhel/glibc-rh1783303-18.patch
      ./rhel/glibc-rh1642150-1.patch
      ./rhel/glibc-rh1642150-2.patch
      ./rhel/glibc-rh1642150-3.patch
      ./rhel/glibc-rh1774115.patch
      ./rhel/glibc-rh1780204-29.patch
      ./rhel/glibc-rh1748197-1.patch
      ./rhel/glibc-rh1748197-2.patch
      ./rhel/glibc-rh1748197-3.patch
      ./rhel/glibc-rh1748197-4.patch
      ./rhel/glibc-rh1748197-5.patch
      ./rhel/glibc-rh1748197-6.patch
      ./rhel/glibc-rh1748197-7.patch
      ./rhel/glibc-rh1642150-4.patch
      ./rhel/glibc-rh1836867.patch
      ./rhel/glibc-rh1821531-1.patch
      ./rhel/glibc-rh1821531-2.patch
      ./rhel/glibc-rh1845098-1.patch
      ./rhel/glibc-rh1845098-2.patch
      ./rhel/glibc-rh1845098-3.patch
      ./rhel/glibc-rh1871387-1.patch
      ./rhel/glibc-rh1871387-2.patch
      ./rhel/glibc-rh1871387-3.patch
      ./rhel/glibc-rh1871387-4.patch
      ./rhel/glibc-rh1871387-5.patch
      ./rhel/glibc-rh1871387-6.patch
      ./rhel/glibc-rh1871394-1.patch
      ./rhel/glibc-rh1871394-2.patch
      ./rhel/glibc-rh1871394-3.patch
      ./rhel/glibc-rh1871395-1.patch
      ./rhel/glibc-rh1871395-2.patch
      ./rhel/glibc-rh1871397-1.patch
      ./rhel/glibc-rh1871397-2.patch
      ./rhel/glibc-rh1871397-3.patch
      ./rhel/glibc-rh1871397-4.patch
      ./rhel/glibc-rh1871397-5.patch
      ./rhel/glibc-rh1871397-6.patch
      ./rhel/glibc-rh1871397-7.patch
      ./rhel/glibc-rh1871397-8.patch
      ./rhel/glibc-rh1871397-9.patch
      ./rhel/glibc-rh1871397-10.patch
      ./rhel/glibc-rh1871397-11.patch
      ./rhel/glibc-rh1880670.patch
      ./rhel/glibc-rh1868106-1.patch
      ./rhel/glibc-rh1868106-2.patch
      ./rhel/glibc-rh1868106-3.patch
      ./rhel/glibc-rh1868106-4.patch
      ./rhel/glibc-rh1868106-5.patch
      ./rhel/glibc-rh1868106-6.patch
      ./rhel/glibc-rh1856398.patch
      ./rhel/glibc-rh1880670-2.patch
      ./rhel/glibc-rh1704868-1.patch
      ./rhel/glibc-rh1704868-2.patch
      ./rhel/glibc-rh1704868-3.patch
      ./rhel/glibc-rh1704868-4.patch
      ./rhel/glibc-rh1704868-5.patch
      ./rhel/glibc-rh1893662-1.patch
      ./rhel/glibc-rh1893662-2.patch
      ./rhel/glibc-rh1855790-1.patch
      ./rhel/glibc-rh1855790-2.patch
      ./rhel/glibc-rh1855790-3.patch
      ./rhel/glibc-rh1855790-4.patch
      ./rhel/glibc-rh1855790-5.patch
      ./rhel/glibc-rh1855790-6.patch
      ./rhel/glibc-rh1855790-7.patch
      ./rhel/glibc-rh1855790-8.patch
      ./rhel/glibc-rh1855790-9.patch
      ./rhel/glibc-rh1855790-10.patch
      ./rhel/glibc-rh1855790-11.patch
      ./rhel/glibc-rh1817513-1.patch
      ./rhel/glibc-rh1817513-2.patch
      ./rhel/glibc-rh1817513-3.patch
      ./rhel/glibc-rh1817513-4.patch
      ./rhel/glibc-rh1817513-5.patch
      ./rhel/glibc-rh1817513-6.patch
      ./rhel/glibc-rh1817513-7.patch
      ./rhel/glibc-rh1817513-8.patch
      ./rhel/glibc-rh1817513-9.patch
      ./rhel/glibc-rh1817513-10.patch
      ./rhel/glibc-rh1817513-11.patch
      ./rhel/glibc-rh1817513-12.patch
      ./rhel/glibc-rh1817513-13.patch
      ./rhel/glibc-rh1817513-14.patch
      ./rhel/glibc-rh1817513-15.patch
      ./rhel/glibc-rh1817513-16.patch
      ./rhel/glibc-rh1817513-17.patch
      ./rhel/glibc-rh1817513-18.patch
      ./rhel/glibc-rh1817513-19.patch
      ./rhel/glibc-rh1817513-20.patch
      ./rhel/glibc-rh1817513-21.patch
      ./rhel/glibc-rh1817513-22.patch
      ./rhel/glibc-rh1817513-23.patch
      ./rhel/glibc-rh1817513-24.patch
      ./rhel/glibc-rh1817513-25.patch
      ./rhel/glibc-rh1817513-26.patch
      ./rhel/glibc-rh1817513-27.patch
      ./rhel/glibc-rh1817513-28.patch
      ./rhel/glibc-rh1817513-29.patch
      ./rhel/glibc-rh1817513-30.patch
      ./rhel/glibc-rh1817513-31.patch
      ./rhel/glibc-rh1817513-32.patch
      ./rhel/glibc-rh1817513-33.patch
      ./rhel/glibc-rh1817513-34.patch
      ./rhel/glibc-rh1817513-35.patch
      ./rhel/glibc-rh1817513-36.patch
      ./rhel/glibc-rh1817513-37.patch
      ./rhel/glibc-rh1817513-38.patch
      ./rhel/glibc-rh1817513-39.patch
      ./rhel/glibc-rh1817513-40.patch
      ./rhel/glibc-rh1817513-41.patch
      ./rhel/glibc-rh1817513-42.patch
      ./rhel/glibc-rh1817513-43.patch
      ./rhel/glibc-rh1817513-44.patch
      ./rhel/glibc-rh1817513-45.patch
      ./rhel/glibc-rh1817513-46.patch
      ./rhel/glibc-rh1817513-47.patch
      ./rhel/glibc-rh1817513-48.patch
      ./rhel/glibc-rh1817513-49.patch
      ./rhel/glibc-rh1817513-50.patch
      ./rhel/glibc-rh1817513-51.patch
      ./rhel/glibc-rh1817513-52.patch
      ./rhel/glibc-rh1817513-53.patch
      ./rhel/glibc-rh1817513-54.patch
      ./rhel/glibc-rh1817513-55.patch
      ./rhel/glibc-rh1817513-56.patch
      ./rhel/glibc-rh1817513-57.patch
      ./rhel/glibc-rh1817513-58.patch
      ./rhel/glibc-rh1817513-59.patch
      ./rhel/glibc-rh1817513-60.patch
      ./rhel/glibc-rh1817513-61.patch
      ./rhel/glibc-rh1817513-62.patch
      ./rhel/glibc-rh1817513-63.patch
      ./rhel/glibc-rh1817513-64.patch
      ./rhel/glibc-rh1817513-65.patch
      ./rhel/glibc-rh1817513-66.patch
      ./rhel/glibc-rh1817513-67.patch
      ./rhel/glibc-rh1817513-68.patch
      ./rhel/glibc-rh1817513-69.patch
      ./rhel/glibc-rh1817513-70.patch
      ./rhel/glibc-rh1817513-71.patch
      ./rhel/glibc-rh1817513-72.patch
      ./rhel/glibc-rh1817513-73.patch
      ./rhel/glibc-rh1817513-74.patch
      ./rhel/glibc-rh1817513-75.patch
      ./rhel/glibc-rh1817513-76.patch
      ./rhel/glibc-rh1817513-77.patch
      ./rhel/glibc-rh1817513-78.patch
      ./rhel/glibc-rh1817513-79.patch
      ./rhel/glibc-rh1817513-80.patch
      ./rhel/glibc-rh1817513-81.patch
      ./rhel/glibc-rh1817513-82.patch
      ./rhel/glibc-rh1817513-83.patch
      ./rhel/glibc-rh1817513-84.patch
      ./rhel/glibc-rh1817513-85.patch
      ./rhel/glibc-rh1817513-86.patch
      ./rhel/glibc-rh1817513-87.patch
      ./rhel/glibc-rh1817513-88.patch
      ./rhel/glibc-rh1817513-89.patch
      ./rhel/glibc-rh1817513-90.patch
      # dl usage information patch, conflicts with Nix makefile changes.
      ./rhel/glibc-rh1817513-91.patch
      ./rhel/glibc-rh1817513-92.patch
      ./rhel/glibc-rh1817513-93.patch
      ./rhel/glibc-rh1817513-94.patch
      ./rhel/glibc-rh1817513-95.patch
      ./rhel/glibc-rh1817513-96.patch
      ./rhel/glibc-rh1817513-97.patch
      ./rhel/glibc-rh1817513-98.patch
      ./rhel/glibc-rh1817513-99.patch
      ./rhel/glibc-rh1817513-100.patch
      ./rhel/glibc-rh1817513-101.patch
      ./rhel/glibc-rh1817513-102.patch
      ./rhel/glibc-rh1817513-103.patch
      ./rhel/glibc-rh1817513-104.patch
      ./rhel/glibc-rh1817513-105.patch
      ./rhel/glibc-rh1817513-106.patch
      ./rhel/glibc-rh1817513-107.patch
      ./rhel/glibc-rh1817513-108.patch
      ./rhel/glibc-rh1817513-109.patch
      ./rhel/glibc-rh1817513-110.patch
      ./rhel/glibc-rh1817513-111.patch
      ./rhel/glibc-rh1817513-112.patch
      ./rhel/glibc-rh1817513-113.patch
      ./rhel/glibc-rh1817513-114.patch
      ./rhel/glibc-rh1817513-115.patch
      ./rhel/glibc-rh1817513-116.patch
      ./rhel/glibc-rh1817513-117.patch
      ./rhel/glibc-rh1817513-118.patch
      ./rhel/glibc-rh1817513-119.patch
      ./rhel/glibc-rh1817513-120.patch
      ./rhel/glibc-rh1817513-121.patch
      ./rhel/glibc-rh1817513-122.patch
      ./rhel/glibc-rh1817513-123.patch
      ./rhel/glibc-rh1817513-124.patch
      ./rhel/glibc-rh1817513-125.patch
      ./rhel/glibc-rh1817513-126.patch
      ./rhel/glibc-rh1817513-127.patch
      ./rhel/glibc-rh1817513-128.patch
      ./rhel/glibc-rh1817513-129.patch
      ./rhel/glibc-rh1817513-130.patch
      ./rhel/glibc-rh1817513-131.patch
      ./rhel/glibc-rh1817513-132.patch
      ./rhel/glibc-rh1882466-1.patch
      ./rhel/glibc-rh1882466-2.patch
      ./rhel/glibc-rh1882466-3.patch
      ./rhel/glibc-rh1817513-133.patch
      ./rhel/glibc-rh1912544.patch
      ./rhel/glibc-rh1918115.patch
      ./rhel/glibc-rh1924919.patch
      ./rhel/glibc-rh1932770.patch
      ./rhel/glibc-rh1936864.patch
      ./rhel/glibc-rh1871386-1.patch
      ./rhel/glibc-rh1871386-2.patch
      ./rhel/glibc-rh1871386-3.patch
      ./rhel/glibc-rh1871386-4.patch
      ./rhel/glibc-rh1871386-5.patch
      ./rhel/glibc-rh1871386-6.patch
      ./rhel/glibc-rh1871386-7.patch
      ./rhel/glibc-rh1912670-1.patch
      ./rhel/glibc-rh1912670-2.patch
      ./rhel/glibc-rh1912670-3.patch
      ./rhel/glibc-rh1912670-4.patch
      ./rhel/glibc-rh1912670-5.patch
      ./rhel/glibc-rh1930302-1.patch
      ./rhel/glibc-rh1930302-2.patch
      ./rhel/glibc-rh1927877.patch
      ./rhel/glibc-rh1918719-1.patch
      ./rhel/glibc-rh1918719-2.patch
      ./rhel/glibc-rh1918719-3.patch
      ./rhel/glibc-rh1934155-1.patch
      ./rhel/glibc-rh1934155-2.patch
      ./rhel/glibc-rh1934155-3.patch
      ./rhel/glibc-rh1934155-4.patch
      ./rhel/glibc-rh1934155-5.patch
      ./rhel/glibc-rh1934155-6.patch
      ./rhel/glibc-rh1956357-1.patch
      ./rhel/glibc-rh1956357-2.patch
      ./rhel/glibc-rh1956357-3.patch
      ./rhel/glibc-rh1956357-4.patch
      ./rhel/glibc-rh1956357-5.patch
      ./rhel/glibc-rh1956357-6.patch
      ./rhel/glibc-rh1956357-7.patch
      ./rhel/glibc-rh1956357-8.patch
      ./rhel/glibc-rh1979127.patch
      ./rhel/glibc-rh1966472-1.patch
      ./rhel/glibc-rh1966472-2.patch
      ./rhel/glibc-rh1966472-3.patch
      ./rhel/glibc-rh1966472-4.patch
      ./rhel/glibc-rh1971664-1.patch
      ./rhel/glibc-rh1971664-2.patch
      ./rhel/glibc-rh1971664-3.patch
      ./rhel/glibc-rh1971664-4.patch
      ./rhel/glibc-rh1971664-5.patch
      ./rhel/glibc-rh1971664-6.patch
      ./rhel/glibc-rh1971664-7.patch
      ./rhel/glibc-rh1971664-8.patch
      ./rhel/glibc-rh1971664-9.patch
      ./rhel/glibc-rh1971664-10.patch
      ./rhel/glibc-rh1971664-11.patch
      ./rhel/glibc-rh1971664-12.patch
      ./rhel/glibc-rh1971664-13.patch
      ./rhel/glibc-rh1971664-14.patch
      ./rhel/glibc-rh1971664-15.patch
      ./rhel/glibc-rh1977614.patch
      ./rhel/glibc-rh1983203-1.patch
      ./rhel/glibc-rh1983203-2.patch
      ./rhel/glibc-rh2021452.patch
      ./rhel/glibc-rh1937515.patch
      ./rhel/glibc-rh1934162-1.patch
      ./rhel/glibc-rh1934162-2.patch
      ./rhel/glibc-rh2000374.patch
      ./rhel/glibc-rh1991001-1.patch
      ./rhel/glibc-rh1991001-2.patch
      ./rhel/glibc-rh1991001-3.patch
      ./rhel/glibc-rh1991001-4.patch
      ./rhel/glibc-rh1991001-5.patch
      ./rhel/glibc-rh1991001-6.patch
      ./rhel/glibc-rh1991001-7.patch
      ./rhel/glibc-rh1991001-8.patch
      ./rhel/glibc-rh1991001-9.patch
      ./rhel/glibc-rh1991001-10.patch
      ./rhel/glibc-rh1991001-11.patch
      ./rhel/glibc-rh1991001-12.patch
      ./rhel/glibc-rh1991001-13.patch
      ./rhel/glibc-rh1991001-14.patch
      ./rhel/glibc-rh1991001-15.patch
      ./rhel/glibc-rh1991001-16.patch
      ./rhel/glibc-rh1991001-17.patch
      ./rhel/glibc-rh1991001-18.patch
      ./rhel/glibc-rh1991001-19.patch
      ./rhel/glibc-rh1991001-20.patch
      ./rhel/glibc-rh1991001-21.patch
      ./rhel/glibc-rh1991001-22.patch
      ./rhel/glibc-rh1929928-1.patch
      ./rhel/glibc-rh1929928-2.patch
      ./rhel/glibc-rh1929928-3.patch
      ./rhel/glibc-rh1929928-4.patch
      ./rhel/glibc-rh1929928-5.patch
      ./rhel/glibc-rh1984802-1.patch
      ./rhel/glibc-rh1984802-2.patch
      ./rhel/glibc-rh1984802-3.patch
      ./rhel/glibc-rh2023420-1.patch
      ./rhel/glibc-rh2023420-2.patch
      ./rhel/glibc-rh2023420-3.patch
      ./rhel/glibc-rh2023420-4.patch
      ./rhel/glibc-rh2023420-5.patch
      ./rhel/glibc-rh2023420-6.patch
      ./rhel/glibc-rh2023420-7.patch
      ./rhel/glibc-rh2033648-1.patch
      ./rhel/glibc-rh2033648-2.patch
      ./rhel/glibc-rh2036955.patch
      ./rhel/glibc-rh2033655.patch
      ./rhel/glibc-rh2007327-1.patch
      ./rhel/glibc-rh2007327-2.patch
      ./rhel/glibc-rh2032281-1.patch
      ./rhel/glibc-rh2032281-2.patch
      ./rhel/glibc-rh2032281-3.patch
      ./rhel/glibc-rh2032281-4.patch
      ./rhel/glibc-rh2032281-5.patch
      ./rhel/glibc-rh2032281-6.patch
      ./rhel/glibc-rh2032281-7.patch
      ./rhel/glibc-rh2045063-1.patch
      ./rhel/glibc-rh2045063-2.patch
      ./rhel/glibc-rh2045063-3.patch
      ./rhel/glibc-rh2045063-4.patch
      ./rhel/glibc-rh2045063-5.patch
      ./rhel/glibc-rh2054790.patch
      ./rhel/glibc-rh2037416-1.patch
      ./rhel/glibc-rh2037416-2.patch
      ./rhel/glibc-rh2037416-3.patch
      ./rhel/glibc-rh2037416-4.patch
      ./rhel/glibc-rh2037416-5.patch
      ./rhel/glibc-rh2037416-6.patch
      ./rhel/glibc-rh2037416-7.patch
      ./rhel/glibc-rh2037416-8.patch
      ./rhel/glibc-rh2033684-1.patch
      ./rhel/glibc-rh2033684-2.patch
      ./rhel/glibc-rh2033684-3.patch
      ./rhel/glibc-rh2033684-4.patch
      ./rhel/glibc-rh2033684-5.patch
      ./rhel/glibc-rh2033684-6.patch
      ./rhel/glibc-rh2033684-7.patch
      ./rhel/glibc-rh2033684-8.patch
      ./rhel/glibc-rh2033684-9.patch
      ./rhel/glibc-rh2033684-10.patch
      ./rhel/glibc-rh2033684-11.patch
      ./rhel/glibc-rh2033684-12.patch
      ./rhel/glibc-rh2063712.patch
      ./rhel/glibc-rh2063042.patch
      ./rhel/glibc-rh2071745.patch
      ./rhel/glibc-rh2065588-1.patch
      ./rhel/glibc-rh2065588-2.patch
      ./rhel/glibc-rh2065588-3.patch
      ./rhel/glibc-rh2065588-4.patch
      ./rhel/glibc-rh2065588-5.patch
      ./rhel/glibc-rh2065588-6.patch
      ./rhel/glibc-rh2065588-7.patch
      ./rhel/glibc-rh2065588-8.patch
      ./rhel/glibc-rh2065588-9.patch
      ./rhel/glibc-rh2065588-10.patch
      ./rhel/glibc-rh2065588-11.patch
      ./rhel/glibc-rh2065588-12.patch
      ./rhel/glibc-rh2065588-13.patch
      ./rhel/glibc-rh2072329.patch
      ./rhel/glibc-rh1982608.patch
      ./rhel/glibc-rh1961109.patch
      ./rhel/glibc-rh2086853.patch
      ./rhel/glibc-rh2077835.patch
      ./rhel/glibc-rh2089247-1.patch
      ./rhel/glibc-rh2089247-2.patch
      ./rhel/glibc-rh2089247-3.patch
      ./rhel/glibc-rh2089247-4.patch
      ./rhel/glibc-rh2089247-5.patch
      ./rhel/glibc-rh2089247-6.patch
      ./rhel/glibc-rh2091553.patch
      ./rhel/glibc-rh1888660.patch
      ./rhel/glibc-rh2096189-1.patch
      ./rhel/glibc-rh2096189-2.patch
      ./rhel/glibc-rh2096189-3.patch
      ./rhel/glibc-rh2080349-1.patch
      ./rhel/glibc-rh2080349-2.patch
      ./rhel/glibc-rh2080349-3.patch
      ./rhel/glibc-rh2080349-4.patch
      ./rhel/glibc-rh2080349-5.patch
      ./rhel/glibc-rh2080349-6.patch
      ./rhel/glibc-rh2080349-7.patch
      ./rhel/glibc-rh2080349-8.patch
      ./rhel/glibc-rh2080349-9.patch
      ./rhel/glibc-rh2047981-1.patch
      ./rhel/glibc-rh2047981-2.patch
      ./rhel/glibc-rh2047981-3.patch
      ./rhel/glibc-rh2047981-4.patch
      ./rhel/glibc-rh2047981-5.patch
      ./rhel/glibc-rh2047981-6.patch
      ./rhel/glibc-rh2047981-7.patch
      ./rhel/glibc-rh2047981-8.patch
      ./rhel/glibc-rh2047981-9.patch
      ./rhel/glibc-rh2047981-10.patch
      ./rhel/glibc-rh2047981-11.patch
      ./rhel/glibc-rh2047981-12.patch
      ./rhel/glibc-rh2047981-13.patch
      ./rhel/glibc-rh2047981-14.patch
      ./rhel/glibc-rh2047981-15.patch
      ./rhel/glibc-rh2047981-16.patch
      ./rhel/glibc-rh2047981-17.patch
      ./rhel/glibc-rh2047981-18.patch
      ./rhel/glibc-rh2047981-19.patch
      ./rhel/glibc-rh2047981-20.patch
      ./rhel/glibc-rh2047981-21.patch
      ./rhel/glibc-rh2047981-22.patch
      ./rhel/glibc-rh2047981-23.patch
      ./rhel/glibc-rh2047981-24.patch
      ./rhel/glibc-rh2047981-25.patch
      ./rhel/glibc-rh2047981-26.patch
      ./rhel/glibc-rh2047981-27.patch
      ./rhel/glibc-rh2047981-28.patch
      ./rhel/glibc-rh2047981-29.patch
      ./rhel/glibc-rh2047981-30.patch
      ./rhel/glibc-rh2047981-31.patch
      ./rhel/glibc-rh2047981-32.patch
      ./rhel/glibc-rh2047981-33.patch
      ./rhel/glibc-rh2047981-34.patch
      ./rhel/glibc-rh2047981-35.patch
      ./rhel/glibc-rh2047981-36.patch
      ./rhel/glibc-rh2047981-37.patch
      ./rhel/glibc-rh2047981-38.patch
      ./rhel/glibc-rh2047981-39.patch
      ./rhel/glibc-rh2047981-40.patch
      ./rhel/glibc-rh2047981-41.patch
      ./rhel/glibc-rh2047981-42.patch
      ./rhel/glibc-rh2047981-43.patch
      ./rhel/glibc-rh2047981-44.patch
      ./rhel/glibc-rh2047981-45.patch
      ./rhel/glibc-rh2047981-46.patch
      ./rhel/glibc-rh2047981-47.patch
      ./rhel/glibc-rh2104907.patch
      ./rhel/glibc-rh2119304-1.patch
      ./rhel/glibc-rh2119304-2.patch
      ./rhel/glibc-rh2119304-3.patch
      ./rhel/glibc-rh2118667.patch
      ./rhel/glibc-rh2122498.patch
      ./rhel/glibc-rh2125222.patch
      ./rhel/glibc-rh1871383-1.patch
      ./rhel/glibc-rh1871383-2.patch
      ./rhel/glibc-rh1871383-3.patch
      ./rhel/glibc-rh1871383-4.patch
      ./rhel/glibc-rh1871383-5.patch
      ./rhel/glibc-rh1871383-6.patch
      ./rhel/glibc-rh1871383-7.patch
      ./rhel/glibc-rh2122501-1.patch
      ./rhel/glibc-rh2122501-2.patch
      ./rhel/glibc-rh2122501-3.patch
      ./rhel/glibc-rh2122501-4.patch
      ./rhel/glibc-rh2122501-5.patch
      ./rhel/glibc-rh2121746-1.patch
      ./rhel/glibc-rh2121746-2.patch
      ./rhel/glibc-rh2116938.patch
      ./rhel/glibc-rh2109510-1.patch
      ./rhel/glibc-rh2109510-2.patch
      ./rhel/glibc-rh2109510-3.patch
      ./rhel/glibc-rh2109510-4.patch
      ./rhel/glibc-rh2109510-5.patch
      ./rhel/glibc-rh2109510-6.patch
      ./rhel/glibc-rh2109510-7.patch
      ./rhel/glibc-rh2109510-8.patch
      ./rhel/glibc-rh2109510-9.patch
      ./rhel/glibc-rh2109510-10.patch
      ./rhel/glibc-rh2109510-11.patch
      ./rhel/glibc-rh2109510-12.patch
      ./rhel/glibc-rh2109510-13.patch
      ./rhel/glibc-rh2109510-14.patch
      ./rhel/glibc-rh2109510-15.patch
      ./rhel/glibc-rh2109510-16.patch
      ./rhel/glibc-rh2109510-17.patch
      ./rhel/glibc-rh2109510-18.patch
      ./rhel/glibc-rh2109510-19.patch
      ./rhel/glibc-rh2109510-20.patch
      ./rhel/glibc-rh2109510-21.patch
      ./rhel/glibc-rh2109510-22.patch
      ./rhel/glibc-rh2109510-23.patch
      ./rhel/glibc-rh2139875-1.patch
      ./rhel/glibc-rh2139875-2.patch
      ./rhel/glibc-rh2139875-3.patch
      ./rhel/glibc-rh1159809-1.patch
      ./rhel/glibc-rh1159809-2.patch
      ./rhel/glibc-rh1159809-3.patch
      ./rhel/glibc-rh1159809-4.patch
      ./rhel/glibc-rh1159809-5.patch
      ./rhel/glibc-rh1159809-6.patch
      ./rhel/glibc-rh1159809-7.patch
      ./rhel/glibc-rh1159809-8.patch
      ./rhel/glibc-rh1159809-9.patch
      ./rhel/glibc-rh1159809-10.patch
      ./rhel/glibc-rh1159809-11.patch
      ./rhel/glibc-rh1159809-12.patch
      ./rhel/glibc-rh2141989.patch
      ./rhel/glibc-rh2142937-1.patch
      ./rhel/glibc-rh2142937-2.patch
      ./rhel/glibc-rh2142937-3.patch
      ./rhel/glibc-rh2144568.patch
      ./rhel/glibc-rh2154914-1.patch
      ./rhel/glibc-rh2154914-2.patch
      ./rhel/glibc-rh2183081-1.patch
      ./rhel/glibc-rh2183081-2.patch
      ./rhel/glibc-rh2172949.patch
      ./rhel/glibc-rh2180155-1.patch
      ./rhel/glibc-rh2180155-2.patch
      ./rhel/glibc-rh2180155-3.patch
      ./rhel/glibc-rh2213909.patch
      ./rhel/glibc-rh2176707-1.patch
      ./rhel/glibc-rh2176707-2.patch
      ./rhel/glibc-rh2186781.patch
      ./rhel/glibc-rh2224348.patch
      ./rhel/glibc-rh2176707-3.patch
      ./rhel/glibc-rh2180462-1.patch
      ./rhel/glibc-rh2180462-2.patch
      ./rhel/glibc-rh2180462-3.patch
      ./rhel/glibc-rh2180462-4.patch
      # Reverted fixes for rh2233338 were here.
      ./rhel/glibc-rh2234714.patch
      ./rhel/glibc-RHEL-2435.patch
      ./rhel/glibc-RHEL-2435-2.patch
      ./rhel/glibc-RHEL-2423.patch
      ./rhel/glibc-RHEL-3036.patch
      ./rhel/glibc-RHEL-3757.patch
      ./rhel/glibc-RHEL-2122.patch
      ./rhel/glibc-RHEL-1192.patch
      ./rhel/glibc-RHEL-3639.patch
      ./rhel/glibc-RHEL-10481.patch
      ./rhel/glibc-RHEL-13720-1.patch
      ./rhel/glibc-RHEL-13720-2.patch
      ./rhel/glibc-RHEL-15867.patch
      ./rhel/glibc-RHEL-16825-1.patch
      ./rhel/glibc-RHEL-16825-2.patch
      ./rhel/glibc-RHEL-16825-3.patch
      ./rhel/glibc-RHEL-16825-4.patch
      ./rhel/glibc-RHEL-15696-1.patch
      ./rhel/glibc-RHEL-15696-2.patch
      ./rhel/glibc-RHEL-15696-3.patch
      ./rhel/glibc-RHEL-15696-4.patch
      ./rhel/glibc-RHEL-15696-5.patch
      ./rhel/glibc-RHEL-15696-6.patch
      ./rhel/glibc-RHEL-15696-7.patch
      ./rhel/glibc-RHEL-15696-8.patch
      ./rhel/glibc-RHEL-15696-9.patch
      ./rhel/glibc-RHEL-15696-10.patch
      ./rhel/glibc-RHEL-15696-11.patch
      ./rhel/glibc-RHEL-15696-12.patch
      ./rhel/glibc-RHEL-15696-13.patch
      ./rhel/glibc-RHEL-15696-14.patch
      ./rhel/glibc-RHEL-15696-15.patch
      ./rhel/glibc-RHEL-15696-16.patch
      ./rhel/glibc-RHEL-15696-17.patch
      ./rhel/glibc-RHEL-15696-18.patch
      ./rhel/glibc-RHEL-15696-19.patch
      ./rhel/glibc-RHEL-15696-20.patch
      ./rhel/glibc-RHEL-15696-21.patch
      ./rhel/glibc-RHEL-15696-22.patch
      ./rhel/glibc-RHEL-15696-23.patch
      ./rhel/glibc-RHEL-15696-24.patch
      ./rhel/glibc-RHEL-15696-25.patch
      ./rhel/glibc-RHEL-15696-26.patch
      ./rhel/glibc-RHEL-15696-27.patch
      ./rhel/glibc-RHEL-15696-28.patch
      ./rhel/glibc-RHEL-15696-29.patch
      ./rhel/glibc-RHEL-15696-30.patch
      ./rhel/glibc-RHEL-15696-31.patch
      ./rhel/glibc-RHEL-15696-32.patch
      ./rhel/glibc-RHEL-15696-33.patch
      ./rhel/glibc-RHEL-15696-34.patch
      ./rhel/glibc-RHEL-15696-35.patch
      ./rhel/glibc-RHEL-15696-36.patch
      ./rhel/glibc-RHEL-15696-37.patch
      ./rhel/glibc-RHEL-15696-38.patch
      ./rhel/glibc-RHEL-15696-39.patch
      ./rhel/glibc-RHEL-15696-40.patch
      ./rhel/glibc-RHEL-15696-41.patch
      ./rhel/glibc-RHEL-15696-42.patch
      ./rhel/glibc-RHEL-15696-43.patch
      ./rhel/glibc-RHEL-15696-44.patch
      ./rhel/glibc-RHEL-15696-45.patch
      ./rhel/glibc-RHEL-15696-46.patch
      ./rhel/glibc-RHEL-15696-47.patch
      ./rhel/glibc-RHEL-15696-48.patch
      ./rhel/glibc-RHEL-15696-49.patch
      ./rhel/glibc-RHEL-15696-50.patch
      ./rhel/glibc-RHEL-15696-51.patch
      ./rhel/glibc-RHEL-15696-52.patch
      ./rhel/glibc-RHEL-15696-53.patch
      ./rhel/glibc-RHEL-15696-54.patch
      ./rhel/glibc-RHEL-15696-55.patch
      ./rhel/glibc-RHEL-15696-56.patch
      ./rhel/glibc-RHEL-15696-57.patch
      ./rhel/glibc-RHEL-15696-58.patch
      ./rhel/glibc-RHEL-15696-59.patch
      ./rhel/glibc-RHEL-15696-60.patch
      ./rhel/glibc-RHEL-15696-61.patch
      ./rhel/glibc-RHEL-15696-62.patch
      ./rhel/glibc-RHEL-15696-63.patch
      ./rhel/glibc-RHEL-15696-64.patch
      ./rhel/glibc-RHEL-15696-65.patch
      ./rhel/glibc-RHEL-15696-66.patch
      ./rhel/glibc-RHEL-15696-67.patch
      ./rhel/glibc-RHEL-15696-68.patch
      ./rhel/glibc-RHEL-15696-69.patch
      ./rhel/glibc-RHEL-15696-70.patch
      ./rhel/glibc-RHEL-15696-71.patch
      ./rhel/glibc-RHEL-15696-72.patch
      ./rhel/glibc-RHEL-15696-73.patch
      ./rhel/glibc-RHEL-15696-74.patch
      ./rhel/glibc-RHEL-15696-75.patch
      ./rhel/glibc-RHEL-15696-76.patch
      ./rhel/glibc-RHEL-15696-77.patch
      ./rhel/glibc-RHEL-15696-78.patch
      ./rhel/glibc-RHEL-15696-79.patch
      ./rhel/glibc-RHEL-15696-80.patch
      ./rhel/glibc-RHEL-15696-81.patch
      ./rhel/glibc-RHEL-15696-82.patch
      ./rhel/glibc-RHEL-15696-83.patch
      ./rhel/glibc-RHEL-15696-84.patch
      ./rhel/glibc-RHEL-15696-85.patch
      ./rhel/glibc-RHEL-15696-86.patch
      ./rhel/glibc-RHEL-15696-87.patch
      ./rhel/glibc-RHEL-15696-88.patch
      ./rhel/glibc-RHEL-15696-89.patch
      ./rhel/glibc-RHEL-15696-90.patch
      ./rhel/glibc-RHEL-15696-91.patch
      ./rhel/glibc-RHEL-15696-92.patch
      ./rhel/glibc-RHEL-15696-93.patch
      ./rhel/glibc-RHEL-15696-94.patch
      ./rhel/glibc-RHEL-15696-95.patch
      ./rhel/glibc-RHEL-15696-96.patch
      ./rhel/glibc-RHEL-15696-97.patch
      ./rhel/glibc-RHEL-15696-98.patch
      ./rhel/glibc-RHEL-15696-99.patch
      ./rhel/glibc-RHEL-15696-100.patch
      ./rhel/glibc-RHEL-15696-101.patch
      ./rhel/glibc-RHEL-15696-102.patch
      ./rhel/glibc-RHEL-15696-103.patch
      ./rhel/glibc-RHEL-15696-104.patch
      ./rhel/glibc-RHEL-15696-105.patch
      ./rhel/glibc-RHEL-15696-106.patch
      ./rhel/glibc-RHEL-15696-107.patch
      ./rhel/glibc-RHEL-15696-108.patch
      ./rhel/glibc-RHEL-15696-109.patch
      ./rhel/glibc-RHEL-15696-110.patch
      ./rhel/glibc-RHEL-17468-1.patch
      ./rhel/glibc-RHEL-17468-2.patch
      ./rhel/glibc-RHEL-19824.patch
      ./rhel/glibc-RHEL-3010-1.patch
      ./rhel/glibc-RHEL-3010-2.patch
      ./rhel/glibc-RHEL-3010-3.patch
      ./rhel/glibc-RHEL-19445.patch
      ./rhel/glibc-RHEL-21997.patch
      ./rhel/glibc-RHEL-31804.patch
      ./rhel/glibc-RHEL-34264.patch
      ./rhel/glibc-RHEL-34267-1.patch
      ./rhel/glibc-RHEL-34267-2.patch
      ./rhel/glibc-RHEL-34273.patch
      ./rhel/glibc-RHEL-52428-1.patch
      ./rhel/glibc-RHEL-52428-2.patch
      ./rhel/glibc-RHEL-39994-1.patch
      ./rhel/glibc-RHEL-39994-2.patch
      ./rhel/glibc-RHEL-36147-1.patch
      ./rhel/glibc-RHEL-36147-2.patch
      ./rhel/glibc-RHEL-36147-3.patch
      ./rhel/glibc-RHEL-49490-1.patch
      ./rhel/glibc-RHEL-49490-2.patch
      ./rhel/glibc-RHEL-61255.patch
      ./rhel/glibc-RHEL-61259-1.patch
      ./rhel/glibc-RHEL-61259-2.patch
      ./rhel/glibc-RHEL-67806.patch
      ./rhel/glibc-RHEL-8381-1.patch
      ./rhel/glibc-RHEL-8381-2.patch
      ./rhel/glibc-RHEL-8381-3.patch
      ./rhel/glibc-RHEL-8381-4.patch
      ./rhel/glibc-RHEL-8381-5.patch
      ./rhel/glibc-RHEL-8381-6.patch
      ./rhel/glibc-RHEL-8381-7.patch
      ./rhel/glibc-RHEL-8381-8.patch
      ./rhel/glibc-RHEL-8381-9.patch
      ./rhel/glibc-RHEL-8381-10.patch
      ./rhel/glibc-RHEL-78390.patch
      ./rhel/glibc-RHEL-83306-1.patch
      ./rhel/glibc-RHEL-83306-2.patch
      ./rhel/glibc-RHEL-35280.patch
      ./rhel/glibc-RHEL-76211.patch
      ./rhel/glibc-RHEL-76387.patch
      ./rhel/glibc-RHEL-86018-1.patch
      ./rhel/glibc-RHEL-86018-2.patch
      ./rhel/glibc-RHEL-88813.patch
      ./rhel/glibc-RHEL-71921.patch
      ./rhel/glibc-RHEL-92685-1.patch
      ./rhel/glibc-RHEL-92685-2.patch
      ./rhel/glibc-RHEL-92685-3.patch
      ./rhel/glibc-RHEL-92685-4.patch
      ./rhel/glibc-RHEL-92685-5.patch
      ./rhel/glibc-RHEL-92685-6.patch
      ./rhel/glibc-RHEL-92685-7.patch
      ./rhel/glibc-RHEL-92685-8.patch
      ./rhel/glibc-RHEL-92685-9.patch
      ./rhel/glibc-RHEL-93937-1.patch
      ./rhel/glibc-RHEL-93937-2.patch
      ./rhel/glibc-RHEL-18039-1.patch
      ./rhel/glibc-RHEL-18039-2.patch
      ./rhel/glibc-RHEL-18039-3.patch
      ./rhel/glibc-RHEL-18039-4.patch
      ./rhel/glibc-RHEL-18039-5.patch
      ./rhel/glibc-RHEL-18039-6.patch
      ./rhel/glibc-RHEL-105326.patch
      ./rhel/glibc-RHEL-114260.patch
      ./rhel/glibc-RHEL-72011-1.patch
      ./rhel/glibc-RHEL-72011-2.patch
      ./rhel/glibc-RHEL-72011-3.patch
      ./rhel/glibc-RHEL-72011-4.patch
      ./rhel/glibc-RHEL-72011-5.patch
      ./rhel/glibc-RHEL-72011-6.patch
      ./rhel/glibc-RHEL-72011-7.patch
      ./rhel/glibc-RHEL-72011-8.patch
      ./rhel/glibc-RHEL-141849.patch
      ./rhel/glibc-RHEL-142194.patch
      ./rhel/glibc-RHEL-142787-1.patch
      ./rhel/glibc-RHEL-142787-2.patch
      ./rhel/glibc-RHEL-24169-1.patch
      ./rhel/glibc-RHEL-24169-2.patch
      ./rhel/glibc-RHEL-24169-3.patch
      ./rhel/glibc-RHEL-24169-4.patch
      ./rhel/glibc-RHEL-24169-5.patch
      ./rhel/glibc-RHEL-24169-6.patch
      ./rhel/glibc-RHEL-24169-7.patch
      ./rhel/glibc-RHEL-24169-8.patch
      ./rhel/glibc-RHEL-24169-9.patch
      ./rhel/glibc-RHEL-24169-10.patch
      ./rhel/glibc-RHEL-24169-11.patch
      ./rhel/glibc-RHEL-24169-12.patch
      ./rhel/glibc-RHEL-24169-13.patch
      ./rhel/glibc-RHEL-24169-14.patch
      ./rhel/glibc-RHEL-24169-15.patch
      ./rhel/glibc-RHEL-24169-16.patch
      ./rhel/glibc-RHEL-24169-17.patch
      ./rhel/glibc-RHEL-24169-18.patch
      ./rhel/glibc-RHEL-24169-19.patch
      ./rhel/glibc-RHEL-24169-20.patch
      ./rhel/glibc-RHEL-24169-21.patch
      ./rhel/glibc-RHEL-24169-22.patch
      ./rhel/glibc-RHEL-137185.patch
      ./rhel/glibc-RHEL-140104.patch

      # Don't use /etc/ld.so.cache, for non-NixOS systems.
      # This applies after the RHEL patches, since they edit files in
      # an incompatibly way with the nixpkgs patch.
      ./dont-use-system-ld-so-cache.patch
    ];

    postPatch = ''
      # Needed for glibc to build with the gnumake 3.82
      # http://comments.gmane.org/gmane.linux.lfs.support/31227
      sed -i 's/ot \$/ot:\n\ttouch $@\n$/' manual/Makefile

      # nscd needs libgcc, and we don't want it dynamically linked
      # because we don't want it to depend on bootstrap-tools libs.
      echo "LDFLAGS-nscd += -static-libgcc" >> nscd/Makefile
    '';

    configureFlags = [
      "-C"
      "--enable-add-ons"
      "--enable-obsolete-nsl"
      "--enable-obsolete-rpc"
      "--sysconfdir=/etc"
      "--enable-stackguard-randomization"
      (lib.withFeatureAs withLinuxHeaders "headers" "${linuxHeaders}/include")
      (lib.enableFeature profilingLibraries "profile")
    ]
    ++ lib.optionals withLinuxHeaders [
      "--enable-kernel=3.2.0" # can't get below with glibc >= 2.26
    ]
    ++ lib.optionals (stdenv.hostPlatform != stdenv.buildPlatform) [
      (lib.flip lib.withFeature "fp" (
        stdenv.hostPlatform.platform.gcc.float or (stdenv.hostPlatform.parsed.abi.float or "hard") == "soft"
      ))
      "--with-__thread"
    ]
    ++ lib.optionals (stdenv.hostPlatform == stdenv.buildPlatform && stdenv.hostPlatform.isAarch32) [
      "--host=arm-linux-gnueabi"
      "--build=arm-linux-gnueabi"

      # To avoid linking with -lgcc_s (dynamic link)
      # so the glibc does not depend on its compiler store path
      "libc_cv_as_needed=no"
    ]
    ++ lib.optional withGd "--with-gd";

    installFlags = [ "sysconfdir=$(out)/etc" ];

    outputs = [
      "out"
      "bin"
      "dev"
      "static"
    ];

    depsBuildBuild = [ buildPackages.stdenv.cc ];
    nativeBuildInputs = [ bison ];
    buildInputs = [
      linuxHeaders
    ]
    ++ lib.optionals withGd [
      gd
      libpng
    ];

    # Needed to install share/zoneinfo/zone.tab.  Set to impure /bin/sh to
    # prevent a retained dependency on the bootstrap tools in the stdenv-linux
    # bootstrap.
    BASH_SHELL = "/bin/sh";

    passthru = { inherit version; };
  }

  // (removeAttrs args [
    "withLinuxHeaders"
    "withGd"
  ])
  //

    {
      name = name + "-${version}${patchSuffix}";

      src = fetchurl {
        url = "mirror://gnu/glibc/glibc-${version}.tar.xz";
        inherit hash;
      };

      # Remove absolute paths from `configure' & co.; build out-of-tree.
      preConfigure = ''
        export PWD_P=$(type -tP pwd)
        for i in configure io/ftwtest-sh; do
            # Can't use substituteInPlace here because replace hasn't been
            # built yet in the bootstrap.
            sed -i "$i" -e "s^/bin/pwd^$PWD_P^g"
        done

        mkdir ../build
        cd ../build

        configureScript="`pwd`/../$sourceRoot/configure"

        ${lib.optionalString (
          stdenv.cc.libc != null
        ) ''makeFlags="$makeFlags BUILD_LDFLAGS=-Wl,-rpath,${stdenv.cc.libc}/lib"''}


      ''
      + lib.optionalString (stdenv.hostPlatform != stdenv.buildPlatform) ''
        sed -i s/-lgcc_eh//g "../$sourceRoot/Makeconfig"

        cat > config.cache << "EOF"
        libc_cv_forced_unwind=yes
        libc_cv_c_cleanup=yes
        libc_cv_gnu89_inline=yes
        EOF
      '';

      preBuild = lib.optionalString withGd "unset NIX_DONT_SET_RPATH";

      doCheck = false; # fails

      meta = {
        homepage = "https://www.gnu.org/software/libc/";
        description = "The GNU C Library";

        longDescription = ''
          Any Unix-like operating system needs a C library: the library which
                  defines the "system calls" and other basic facilities such as
                  open, malloc, printf, exit...

                  The GNU C library is used as the C library in the GNU system and
                  most systems with the Linux kernel.
        '';

        license = lib.licenses.lgpl2Plus;

        maintainers = [ lib.maintainers.eelco ];
        platforms = lib.platforms.linux;
      }
      // meta;
    }

  // lib.optionalAttrs (stdenv.hostPlatform != stdenv.buildPlatform) {
    preInstall = null; # clobber the native hook

    # To avoid a dependency on the build system 'bash'.
    preFixup = ''
      rm -f $bin/bin/{ldd,tzselect,catchsegv,xtrace}
    '';
  }
)
