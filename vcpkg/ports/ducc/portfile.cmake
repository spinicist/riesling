vcpkg_from_gitlab(
    GITLAB_URL https://gitlab.mpcdf.mpg.de/
    OUT_SOURCE_PATH SOURCE_PATH
    REPO mtr/ducc
    REF 7632141550057953c9d20c99d68e92dec21554e6
    SHA512 6e6f96d99b88fcf68e710bd2aa0a19c3e4fd6f98bc4cfa50a385083ae6988b97712b7e4e433281f360b42695727494df00a46f6ff3a971e6800e8b755cd5b0c2
    HEAD_REF master
    PATCHES
)

file(COPY "${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt" DESTINATION "${SOURCE_PATH}")

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME ducc)
# copy license
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME copyright)

vcpkg_copy_pdbs()
