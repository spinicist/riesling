vcpkg_buildpath_length_warning(37)

block(SCOPE_FOR VARIABLES PROPAGATE SOURCE_PATH)
set(VCPKG_BUILD_TYPE release) # header-only

vcpkg_from_gitlab(
    GITLAB_URL https://gitlab.com
    OUT_SOURCE_PATH SOURCE_PATH
    REPO libeigen/eigen
    REF af59ada0accc722ebd930ec65340156dbe62ffa1
    SHA512 e8de25f3e1760ee48dee30a25a869cdb584303a2cd18cbcb00f23c922d7786ab5a281d1c20db61617aa95dd932054461e2d2459c95bbf332c719a6b1b4acf4c9
    HEAD_REF master
    PATCHES
)

# vcpkg_from_git(
#     URL file:///Users/tobias/Code/eigen
#     OUT_SOURCE_PATH SOURCE_PATH
#     REF 78cb63a5d465d708be850c011d450d92d44d8434
# )

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTING=OFF
        -DEIGEN_BUILD_DOC=OFF
        -DEIGEN_BUILD_PKGCONFIG=ON
        -DEIGEN_BUILD_BLAS=OFF
        -DEIGEN_BUILD_LAPACK=OFF
        -DEIGEN_BUILD_DEMOS=OFF
        "-DCMAKEPACKAGE_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/share/eigen3"
        "-DPKGCONFIG_INSTALL_DIR=${CURRENT_PACKAGES_DIR}/lib/pkgconfig"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup()
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/eigen3/Eigen3Config.cmake" "if (NOT TARGET eigen)" "if (NOT TARGET Eigen3::Eigen)")
endblock()

if(NOT VCPKG_BUILD_TYPE)
    file(INSTALL "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/eigen3.pc" DESTINATION "${CURRENT_PACKAGES_DIR}/debug/lib/pkgconfig")
endif()
vcpkg_fixup_pkgconfig()

file(GLOB INCLUDES "${CURRENT_PACKAGES_DIR}/include/eigen3/*")
# Copy the eigen header files to conventional location for user-wide MSBuild integration
file(COPY ${INCLUDES} DESTINATION "${CURRENT_PACKAGES_DIR}/include")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/COPYING.README")
