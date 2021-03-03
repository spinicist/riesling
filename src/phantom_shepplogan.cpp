#include "phantom_shepplogan.h"
#include <stdio.h>

/* Creates a 3D Shepp Logan using the method outlined in
* Cheng, G. K., Sarlls, J. E., & Özarslan, E. (2007). 
* Three-dimensional analytical magnetic resonance imaging phantom in the Fourier domain. 
* Magnetic Resonance in Medicine, 58(2), 430–436. https://doi.org/10.1002/mrm.21292
*
* Here we calculate the phantom in image space instead of k-space though to use with sense maps
*
* Doesn't support the off-centre option or size scaling
*/

Cx3 SheppLoganPhantom(Info const &info, float const intensity, Log const &log)
{
    log.info(FMT_STRING("Drawing 3D Shepp Logan Phantom. Intensity {}"), intensity);
    Cx3 phan(info.matrix[0], info.matrix[1], info.matrix[2]);
    phan.setZero();

    // Parameters for the 10 elipsoids in the 3D Shepp-Logan phantom from Cheng et al.
    Eigen::ArrayXXf centres(10,3);
    centres <<  0,      0,      0,           
                0,      0,      0,           
               -0.22,   0,     -0.25,
                0.22,   0,     -0.25,
                0,      0.35,  -0.25,    
                0,      0.1,   -0.25,     
               -0.08,  -0.65,  -0.25,
                0.06,  -0.65,  -0.25,
                0.06,  -0.105,  0.625,
                0,      0.1,    0.625;
    
    Eigen::ArrayXXf ha(10,3); // Half-axes
    ha <<   0.69,   0.92,   0.9,    
            0.6624, 0.874,  0.88,
            0.41,   0.16,   0.21,   
            0.31,   0.11,   0.22,   
            0.21,   0.25,   0.5,    
            0.046,  0.046,  0.046,
            0.046,  0.023,  0.02, 
            0.046,  0.023,  0.02, 
            0.056,  0.04,   0.1,   
            0.056,  0.056,  0.1;
                
    Eigen::ArrayXXf angles(1,10);
    angles << 0, 0, 3*M_PI/5, 2*M_PI/5, 0, 0, 0, M_PI/2, M_PI/2, 0;

    Eigen::ArrayXXf pd(1,10);
    pd << 2, -0.8, -0.2, -0.2, 0.2, 0.2, 0.1, 0.1, 0.2, -0.2;

    // Scale the default parameters by the user input
    pd *= intensity/pd.maxCoeff();

    long const cx = phan.dimension(0) / 2;
    long const cy = phan.dimension(1) / 2;
    long const cz = phan.dimension(2) / 2;
    
    // Here we use normalised coordinates between -1 and 1 in rx,ry,rz
    for (long iz = 0; iz < phan.dimension(2); iz++){
        auto const rz = (1.0*iz)/cz - 1;
        for (long iy = 0; iy < phan.dimension(1); iy++) {
            auto const ry = (1.0*iy)/cy - 1;
            for (long ix = 0 ; ix < phan.dimension(0); ix++) {
                auto const rx = (1.0*ix)/cx - 1;
                
                // Loop over the 10 elipsoids
                for (long ie = 0; ie < 10; ie++){
                    auto const px = (rx - centres(ie,0))*cos(angles(ie)) + (ry - centres(ie,1))*sin(angles(ie));
                    auto const py = (rx - centres(ie,0))*sin(angles(ie)) - (ry - centres(ie,1))*cos(angles(ie));
                    auto const pz = rz - centres(ie,2);
                    auto const r = px*px/(ha(ie,0)*ha(ie,0)) + py*py/(ha(ie,1)*ha(ie,1)) + pz*pz/(ha(ie,2)*ha(ie,2));

                    if ((r) < 1) {    
                        phan(ix,iy,iz) += pd(ie);
                    } 

                }
            }
        }
    }

    log.image(phan, "phantom-shepplogan.nii");
    return phan;
}