import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import h5py
import os
plt.style.use('dark_background')


def load_riesling(f):
    data = abs(np.squeeze(nib.load(f).get_fdata(dtype=np.complex64)))
    data = data/np.quantile(data[...], 0.95)
    data = np.rot90(data, 2)

    return data


def load_rrsg(f):
    h5 = h5py.File(f, 'r')
    data = h5['CG_reco'][...]
    h5.close()

    # Interested in last iteration
    img = abs(data[10, :, :])
    img = img/np.quantile(img[...], 0.95)
    img = np.rot90(img, 2)

    return img


def make_comparison(riesling_img, rrsg_img, fname=None, vmax=1.2, dmax=100):
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize=(14, 4))
    fig.add_subplot(1, 3, 1)
    plt.imshow(riesling_img, cmap='gray', vmin=0, vmax=vmax)
    plt.axis('off')
    plt.title('riesling')

    fig.add_subplot(1, 3, 2)
    plt.imshow(rrsg_img, vmin=0, vmax=vmax, cmap='gray')
    plt.axis('off')
    plt.title('CG-SENSE Python Ref')

    fig.add_subplot(1, 3, 3)
    plt.imshow((rrsg_img - riesling_img)/rrsg_img *
               100, vmin=-dmax, vmax=dmax, cmap='gray')
    plt.axis('off')
    cb = plt.colorbar(label='Relative difference in %')
    plt.title('Relative difference')

    plt.tight_layout()

    if fname:
        plt.savefig(fname, dpi=300)

    plt.show()


def main():
    # riesling recon
    riesling_challenge = load_riesling(
        'riesling_recon/rrsg_challenge_brain_toe10.nii')
    riesling_reference = load_riesling(
        'riesling_recon/rrsg_reference_brain_toe10.nii')

    # Challenge recon
    rrsg_challenge = load_rrsg(
        'rrsg_recon/challenge/output/python/brain/CG_reco_inscale_True_denscor_True_reduction_1.h5')
    rrsg_reference = load_rrsg(
        'rrsg_recon/challenge/output/python/heart/CG_reco_inscale_True_denscor_True_reduction_1.h5')

    make_comparison(riesling_challenge, rrsg_challenge,
                    'challenge_brain_comparison.png')

    make_comparison(riesling_reference, rrsg_reference,
                    'reference_brain_comparison.png')


if __name__ == "__main__":
    main()
