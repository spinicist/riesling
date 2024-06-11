
vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO mreineck/ducc
    REF 95255f5efdd33e4406d4be19a59d5ee0f1a4d5ba
    SHA512 a8239c4aa4dfa75f4e3c2ce86fdf260ff685d842b8f72a4c1fc5a838911c38f268e4c42a162dbf1cdd433f60f71556728a6f9c4d0ba251f04ac4424f75cb9212
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
