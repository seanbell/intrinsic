import os
import numpy as np
from scipy.misc import imread, imsave
from scipy.ndimage import gaussian_filter
from skimage.filter import denoise_bilateral

# relative luminance for sRGB:
RGB_TO_Y = np.array([0.2126, 0.7152, 0.0722])


def load(filename, is_srgb=True):
    if not filename:
        raise ValueError("Empty filename")
    image = imread(filename).astype(np.float) / 255.0
    if is_srgb:
        return srgb_to_rgb(image)
    else:
        return image


def load_mask(filename):
    if not filename:
        raise ValueError("Empty filename")
    image = imread(filename)
    if image.ndim == 2:
        return (image >= 128)
    elif image.ndim == 3:
        return (np.mean(image, axis=-1) >= 128)
    else:
        raise ValueError("Unknown mask format")


def save(filename, image, mask_nz=None, rescale=False, srgb=True):
    """ (Optionally) Remap to sRGB and save """
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)
    if rescale:
        image = rescale_for_display(image, mask_nz=mask_nz)

    if mask_nz is not None:
        image2 = np.zeros_like(image)
        if srgb:
            image2[mask_nz] = rgb_to_srgb(image[mask_nz]) * 255.0
        else:
            image2[mask_nz] = image[mask_nz] * 255.0
    else:
        if srgb:
            image2 = rgb_to_srgb(image) * 255.0
        else:
            image2 = image * 255.0

    assert not np.isnan(image2).any()
    image2[image2 > 255] = 255
    image2[image2 < 0] = 0
    imsave(filename, image2.astype(np.uint8))


def gray_to_rgb(gray):
    rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=gray.dtype)
    rgb[:, :, :] = gray[:, :, np.newaxis]
    return rgb


def luminance(image):
    """ Returns the luminance image """
    if image.ndim == 2:
        return np.dot(RGB_TO_Y, image.T).T
    else:
        rows, cols, _ = image.shape
        image_flat = image.reshape(rows * cols, 3)
        Y_flat = np.dot(RGB_TO_Y, image_flat.T).T
        return Y_flat.reshape(image.shape[0:2])


def rescale_for_display(image, mask_nz=None, percentile=99.9):
    """ Rescales an image so that a particular perenctile is mapped to pure
    white """
    if mask_nz is not None:
        return image / np.percentile(image, percentile)
    else:
        return image / np.percentile(image[mask_nz], percentile)


def rgb_to_srgb(rgb):
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret


def srgb_to_rgb(srgb):
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb_to_irg(rgb):
    """ converts rgb to (mean of channels, red chromaticity, green chromaticity) """
    irg = np.zeros_like(rgb)
    s = np.sum(rgb, axis=-1)
    irg[..., 0] = s / 3.0
    irg[..., 1] = rgb[..., 0] / s
    irg[..., 2] = rgb[..., 1] / s
    return irg


def irg_to_rgb(irg):
    """ converts (mean of channels, red chromaticity, green chromaticity) to rgb """
    rgb = np.zeros_like(irg)
    s = irg[..., 0] * 3.0
    rgb[..., 0] = irg[..., 1] * s
    rgb[..., 1] = irg[..., 2] * s
    rgb[..., 2] = (1.0 - irg[..., 1] - irg[..., 2]) * s

    #np.testing.assert_array_almost_equal(
        #irg, rgb_to_irg(rgb))

    return rgb


def gaussian_blur_gray_image_nz(image_nz, image_shape, mask_nz, sigma):
    """ Blur a masked grayscale image """

    # deal with the mask -- set the unmasked entries to the average
    orig_mean = np.mean(image_nz)
    image = np.empty(image_shape[0:2])
    image.fill(orig_mean)
    image[mask_nz] = image_nz

    blurred = gaussian_filter(image, sigma=sigma)
    blurred_nz = blurred[mask_nz]

    # adjust to keep the mean the same
    new_mean = np.mean(blurred_nz)
    blurred_nz *= orig_mean / new_mean
    return blurred_nz


def bilateral_blur_gray_image_nz(image_nz, image_shape, mask_nz, sigma_range, sigma_spatial):
    """ Blur a masked grayscale image """

    # deal with the mask -- set the unmasked entries to the average
    orig_mean = np.mean(image_nz)
    image = np.empty(image_shape[0:2])
    image.fill(orig_mean)
    image[mask_nz] = image_nz

    blurred = denoise_bilateral(
        image,
        sigma_range=sigma_range,
        sigma_spatial=sigma_spatial,
        win_size=max(int(sigma_spatial * 2), 3),
        mode='reflect',
    )
    blurred_nz = blurred[mask_nz]

    # adjust to keep the mean the same
    new_mean = np.mean(blurred_nz)
    blurred_nz *= orig_mean / new_mean
    return blurred_nz


def n_distinct_colors(n):
    # return choices from well-distributed precomputed colors
    if n <= DISTINCT_COLORS.shape[0]:
        return DISTINCT_COLORS[:n, :]

    # add random colors for remaining
    return np.vstack([
        DISTINCT_COLORS,
        np.random.rand(n - DISTINCT_COLORS.shape[0], 3)
    ])

# start off with some distinct colors
DISTINCT_COLORS = np.array(
    [(int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)) for s in [
        "ff0000", "ff6c0f", "9d9158", "345f1d", "3b7e64", "aeefff", "57adff",
        "00093b", "ba62ff", "520047", "ff84ba", "ff8484", "ff8031", "fff4b5",
        "050f00", "00ad6b", "00aad6", "8cc7ff", "00062b", "6400b0", "46003c",
        "f7529a", "ff8f8e", "ffb78b", "fbf5cd", "acff81", "002b1a", "008fb4",
        "6db6f9", "6276ff", "520090", "3e0536", "f30069", "ee0000", "fa5e00",
        "d9d3ac", "9fe47b", "006f46", "0081a0", "0076e6", "001bc4", "3c006a",
        "f344d9", "c2427a", "ee0000", "e5a277", "a89000", "7dc25c", "00462c",
        "367b8c", "6595c1", "0016a2", "e2b9ff", "932283", "960042", "ed0000",
        "c28259", "a59024", "5da13e", "1a6047", "004151", "0056a9", "000e6f",
        "8b00e7", "ff1cd9", "813f5c", "d9a3a3", "c34d00", "867200", "1e6200",
        "78e5c1", "00222b", "003769", "b4bdff", "9d03ff", "ffa7f1", "5b0027",
        "d30000", "9f643c", "807222", "0d2900", "00cf89", "daf7ff", "00284c",
        "2f41d2", "61009d", "ff68e5", "e66399", "d10000", "ffc296", "7d723b",
        "93ff65", "003523", "00c8ff", "0081ff", "dbdfff", "30004d", "b546a2",
        "b2517a", "d20000", "ff7006", "ffe225", "89e861", "00f6a9", "64deff",
        "86b5e3", "000d82", "bf54ff", "36002c", "b1004b", "b90000", "8f3c00",
        "ffeb6b", "288400", "55c3a1", "79bacd", "005ebd", "5a65df", "f3ddff",
        "ffb3f1", "a25d7a", "b78383", "7e4721", "ccb000", "3c811e", "009162",
        "589aac", "004588", "000d95", "d78dff", "df00b7", "ff4691", "a00000",
        "652a00", "b7b18c", "48f106", "006746", "006681", "002f5d", "2c3276",
        "d07aff", "d867c3", "ff3886", "920000", "3b1900", "97926d", "66c640",
        "4dffca", "003646", "0077f4", "00085d", "ac26f5", "ba0097", "e59bb9",
        "860000", "fd9749", "5e551f", "75ff41", "9affe3", "9adcef", "0064d1",
        "2a37e7", "a73ce0", "5b2452", "d57099", "790000", "ffb980", "fff39e",
        "6eed42", "2ea281", "00b5eb", "0055af", "9095ff", "8800d2", "fb87e5",
        "c37b99", "6e0000", "ffae69", "ffea4d", "42a51d", "008963", "a5b3b8",
        "004c9c", "1e2bfc", "51007d", "a30083", "be004c", "620000", "ff8c2e",
        "787350", "042700", "000d09", "0089b5", "003976", "00056f", "3a0058",
        "960077", "a30043", "5b0000", "d8782a", "605500", "89ff81", "00b080",
        "007ca1", "00244c", "7e83ff", "ca5fff", "7a406f", "cc004d", "540000",
        "602b00", "fff385", "a2ff9b", "003024", "859397", "00152b", "4c4d95",
        "c26bed", "ffeffb", "660027", "4e0000", "381900", "5a5634", "011400",
        "00251b", "007294", "000e1c", "a2a2ff", "a04bcb", "ffbdf0", "4e001e",
        "440000", "ffa151", "403a02", "7fe97b", "43eac1", "006281", "004089",
        "5d5df5", "8411be", "bb7cad", "e70055", "3a0000", "ffe5ce", "fff26a",
        "69ff64", "00d69f", "677478", "002958", "0000b6", "7e2aaa", "9a5e8d",
        "d62868", "350000", "efcbac", "3d3a1a", "b7ffb4", "00a681", "004c62",
        "001b3b", "000082", "c441ff", "800064", "800030", "2f0000", "ea6c00",
        "292500", "64ed60", "008263", "4a575b", "5da7ff", "000082", "7700b1",
        "720059", "ff98bb", "280000", "ccaa8d", "242000", "07ce00", "00ffca",
        "2f3b3f", "0070f4", "00005e", "783896", "ff9ae8", "ff75a5", "240000",
        "ab8b6e", "fff21c", "95e493", "00cca0", "002935", "0058c3", "00003c",
        "5d0089", "dd9cce", "ff86b0", "1d0000", "ffb672", "090800", "5cc65b",
        "6effe2", "162125", "004eaf", "00002c", "591976", "d600a3", "ffa5c5",
        "180000", "e9a160", "fef969", "00f800", "00efc0", "00c1ff", "00469c",
        "6c6ab6", "b100ff", "c631a3", "f60056", "ffb0ae", "c58242", "fff94a",
        "74c274", "004c3e", "e7f6fb", "00387c", "2e2b89", "dd95fa", "281c25",
        "a31a4a", "fcc4c3", "8a6c51", "ddd71e", "00b100", "00221b", "00afeb",
        "003069", "1e1d35", "ba75d8", "26001d", "8c0030", "ff4038", "6b4f35",
        "dad747", "00b100", "008f77", "c5d4d9", "d4e6ff", "c6c3ff", "9956b6",
        "ff75dd", "730c2f", "ffe6e5", "ff8202", "b9b600", "00af00", "006c5a",
        "00a1d6", "006dff", "5e55ff", "1f002c", "ff12c4", "720028", "ff4e47",
        "ffdbb7", "f9fa84", "00ac00", "002b24", "002c3b", "a6cbff", "8c88d7",
        "532664", "5f515b", "58001f", "f3605c", "ffd0a1", "b7b622", "008f00",
        "00d7b5", "72d9ff", "e7f1ff", "04004c", "350947", "5f0047", "ffb9d0",
        "ffa8a4", "ff972c", "969600", "008a00", "00b195", "00bbff", "78b1ff",
        "bab5ff", "eab1ff", "51003d", "20000b", "ffd1cf", "ffc48a", "d6d864",
        "008800", "00d1b6", "4fcfff", "91c0ff", "847bff", "de83ff", "43353f",
        "fb5186", "a9150f", "b45900", "b3b644", "008800", "008a77", "009cd6",
        "5191ec", "ada8fa", "d04fff", "ea57c3", "b30039", "68201d", "a26325",
        "747700", "006a00", "00675a", "0095c9", "0053c3", "1200d9", "9c00d3",
        "decdd9", "990031", "ff968d", "4d341b", "565a00", "006c00", "00ffe1",
        "005e81", "003a8f", "383651", "74009d", "bcacb8", "7d0029", "ff8179",
        "3a1d00", "3c3e00", "006a00", "00ab96", "004863", "00347c", "02001d",
        "714282", "b20083", "ff6090", "ff7164", "341a00", "919622", "006800",
        "00463e", "00a7ec", "002e6f", "1100a3", "4e006a", "9c8c98", "ff709b",
        "ff5d4f", "190d00", "efff0f", "004e00", "60e6d8", "389bc1", "002558",
        "2e219c", "d39de4", "7d6e79", "ff447c", "de2312", "ffc27f", "f3fa9d",
        "004700", "8ae1d8", "0082b5", "031e46", "54526e", "b17ec2", "6a1152",
        "ff3672", "8a3d36", "ffb868", "cadd00", "003400", "00c9b6", "006a94",
        "5fa0ff", "5e4aff", "915fa2", "e600a3", "c1003a", "ffa399", "ffad4f",
        "d0d87d", "002c00", "35c4b7", "003246", "0065fb", "918dad", "2e003c",
        "8e0065", "a50032", "ac5b52", "ff8c00", "202400", "002300", "69bfb7",
        "002635", "005de6", "5146a9", "c000f6", "4b0036", "460216", "9a2f24",
        "ffa639", "ecff48", "002000", "002824", "001e2b", "0053d2", "726f8d",
        "61007d", "ff89dc", "ea0042", "260400", "f5cb96", "aeb75e", "001c00",
        "00ebd7", "00141c", "003d9c", "a99bff", "e577ff", "ff66d0", "ce003b",
        "0c0100", "eca148", "8d973f", "001c00", "00a396", "00b4ff", "295495",
        "d2cef0", "d63aff", "fb40c4", "c74067", "ff3015", "db7700", "6d7821",
        "001700", "479e97", "60bbe3", "b3d0ff", "b1adce", "ba1ee0", "40002d",
        "520017", "cf3e2a", "814600", "a4bc00", "001600", "008378", "008ec9",
        "276ddf", "8873ff", "9700be", "ff04af", "ff83a4", "ce796f", "371e00",
        "4f5a00", "001800", "001e1b", "005275", "004bc4", "7363ca", "6f008a",
        "c00084", "f80043", "be4e3e", "311b00", "d8ff06", "001800", "00fced",
        "004563", "0045b0", "2b13b0", "3d004d", "8b336f", "2b000b", "ab1400",
        "ff9200", "e7ff68", "001000", "227f78", "d6f2ff", "003281", "9481ed",
        "de4bff", "780052", "ffb1c4", "ff9483", "ffb657", "c4dd46", "000b00",
        "00605a", "b6e8ff", "001d4c", "2300c4", "2f1536", "ac528e", "ffdce5",
        "ff4d30", "d2aa76", "f2f5e4", "000200", "003834", "54c9ff", "00173b",
        "553ebd", "f1a7ff", "a90070", "eb6285", "ff6d56", "c78128", "d0d3c2",
        "3aca3f", "85fffa", "00aeff", "e0ecff", "8b69ff", "b338cb", "6b0048",
        "d92654", "e36e5b", "b08a58", "afb2a2", "54a156", "acfffa", "00a3ec",
        "0056e6", "eae4ff", "694a6f", "ffbae7", "95304a", "ffbaae", "593200",
        "7f9c00", "c9ffcc", "ccfdfb", "0096de", "004ed8", "785be0", "4a0058",
        "ff99db", "5d0017", "ff8e79", "8f6c3c", "8f9282", "39f140", "aadbd8",
        "007db5", "003ba2", "1f008f", "4b2f51", "dc008f", "ff96af", "ff836d",
        "341e00", "717464", "37a53d", "00d8cb", "002f46", "00276a", "b6a1ff",
        "d75cee", "ce72ae", "680018", "f55f46", "ffad2f", "535648", "a7dfab",
        "8ab9b7", "7dd3ff", "00235d", "5733d2", "c8a5ce", "9c0065", "34000c",
        "e02100", "ffc167", "1f2600", "87be8b", "6b9997", "00a9ff", "00030a",
        "0e003c", "a785ad", "9a1f70", "b75066", "210500", "ffb84d", "b3e200",
        "689d6c", "00928b", "cbedff", "0059fc", "9b79ff", "9104aa", "0f000a",
        "b50026", "ff8163", "ffcf89", "323e00", "338139", "00d1cc", "0087c9",
        "4d71b5", "af92ff", "85009d", "ff4abb", "9a0021", "ff3000", "eea029",
        "383a2d", "497e4f", "00b2ab", "0070a8", "002d81", "cebbff", "87678d",
        "eb0090", "8e0020", "ffc5b9", "a46300", "ecfbb5", "2b5f33", "4c7a78",
        "006395", "000f2b", "7d51f5", "ee80ff", "cc2e8f", "7f001a", "4a0d00",
        "6e4f20", "cad994", "008b1b", "2f5c5b", "004d75", "80abff", "5723e7",
        "ea68ff", "bc448e", "740019", "ff5d36", "ffdaa0", "a8b775", "7cff9a",
        "002b2a", "004263", "004ee7", "231747", "eac6f0", "b70070", "ff5071",
        "ff714d", "d6a95f", "617c00", "59ff80", "00fff9", "00a4ff", "0043c4",
        "e2d7ff", "5d006b", "860053", "ff0034", "ffaf98", "533300", "1e2114",
        "006e19", "00f0ee", "008fde", "003eb6", "1b005e", "a800be", "570037",
        "ff345e", "1b0600", "ffe4b7", "d4ff46", "08631c", "006e6d", "002d46",
        "003394", "f4f0ff", "6a0076", "ff86cf", "ff7390", "ff9575", "ffa000",
        "e2ff83", "00f83e", "0f403f", "002235", "00226a", "25007c", "ac48b6",
        "ffa6db", "ffa9ba", "ffa88c", "ffb535", "889757", "55ed7a", "002525",
        "91d7ff", "001f5d", "403063", "8a2696", "ff73c7", "eb002e", "ffb7a3",
        "ffedcd", "beff00", "72e993", "62ffff", "007eca", "0052fc", "0d002c",
        "1a001d", "ff009b", "db6f84", "9b2e0b", "c98100", "bfde63", "95ffb3",
        "bcfeff", "006ba8", "6f90d7", "8045ff", "f189fa", "e065ae", "cf0029",
        "421000", "b48a41", "90c100", "00d53d", "00faff", "005f95", "0032a2",
        "5501fc", "cd00e1", "770048", "c20028", "ff3e00", "301f00", "9dbc43",
        "d9fce3", "00c9cc", "003757", "00194c", "4800d9", "ce69d8", "4b002d",
        "a70022", "f75f2e", "ffc14c", "7c9c20", "b7dac2", "00a8ab", "0095f3",
        "598bff", "7c68a1", "7f008a", "30001d", "66212e", "e7ccc3", "fec966",
        "69783a", "4ec773", "9afbff", "0089de", "91aff9", "5d4b82", "f22cff",
        "ff3daf", "3d000c", "e56d45", "926b24", "4b5b1e", "97b9a1", "00d9e1",
        "0074bc", "0044e7", "260d58", "a200aa", "ffc7e6", "ff90a4", "e0a28d",
        "f1a000", "cfff67", "789882", "00878c", "1378b5", "002882", "b389ff",
        "c614cb", "ff64bb", "ff5f7b", "d2795a", "daa946", "9ae800", "004b18",
        "004d50", "005388", "aec5ff", "a170ff", "4a004d", "c50071", "46000d",
        "d03d0e", "ffac00", "6aa000", "001908", "00292a", "003d63", "0048fc",
        "bca6e4", "3a003c", "930053", "ffc6d0", "c5aba2", "ffda88", "3f5f00",
        "23cb5b", "00686d", "001a2b", "0031b0", "9b87c2", "2b002c", "ff99d2",
        "ff859a", "c04e28", "b68926", "ace344", "00b23a", "38f3ff", "0079ca",
        "263763", "be99ff", "fc63ff", "e1007c", "5a000f", "bd836e", "724e00",
        "b9ff44", "5a7964", "7df7ff", "0065a8", "011958", "3b00a3", "ea47ee",
        "630037", "50000e", "af5a3d", "322200", "a1ff00", "3d5b47", "00b3c0",
        "005a95", "00113b", "452876", "631564", "ffb2da", "ff6f85", "a48b83",
        "ffc231", "dbff9c", "003512", "003135", "004675", "0039d8", "f2eaff",
        "580059", "ff0087", "db2541", "9b6451", "ffc94a", "b9de7c", "00a93c",
        "00cfe2", "002d4b", "0035c4", "8135ff", "ff9eff", "ad005d", "ca3f53",
        "8c3d22", "ffe39f", "98bd5d", "004619", "209fac", "00233b", "2e4fa9",
        "4b00c4", "ffe9ff", "55002d", "863e49", "856d65", "ffb700", "779c3e",
        "89e5aa", "00939f", "0097ff", "002ca2", "644495", "ffb6ff", "430326",
        "ffbbc4", "7b4736", "dda927", "587d20", "223f2d", "00808c", "85cdff",
        "002894", "491d89", "ffcbff", "ff48a6", "ff97a4", "665048", "ffd96e",
        "274200", "aaffcb", "00191c", "006fbc", "001e6f", "1f004d", "ff88ff",
        "ff87c6", "a75c66", "5b2c1c", "ba8900", "c8ff82", "00f95e", "00e6ff",
        "004f88", "5e85ff", "dec7ff", "ffd5ff", "ef007c", "ffa4af", "49342d",
        "946b00", "172900", "67c38b", "4ec0cd", "1f5981", "87a4ff", "a665ff",
        "ffe1ff", "a00054", "ef6170", "ffdcce", "ffc000", "a6e362", "003013",
        "00adc0", "1e3c51", "002182", "a57fd7", "ffdbff", "7d0041", "ec9aa3",
        "ffc3ad", "ffecb6", "63a11f", "008b37", "a5f5ff", "008fff", "00134c",
        "8461b6", "ffbeff", "6f0e40", "c97a84", "ffa282", "e0a800", "458100",
        "00d559", "87f1ff", "50b2ff", "003cfc", "2e0070", "ff70ff", "3a001e",
        "a80013", "ff8254", "4c3900", "7bec00", "46a26c", "008ea0", "0076d1",
        "002bb6", "c69ffa", "ff93ff", "361325", "ff828f", "ff946c", "2e2200",
        "84c142", "072514", "007180", "0061a9", "455381", "4c00b0", "833582",
        "ff76ba", "fa0019", "2d1b15", "ffe387", "132400", "001308", "00616e",
        "004375", "00165d", "6a3aa9", "6b006b", "bb005e", "c30014", "ff804c",
        "ffce2e", "b3ff66", "3fff9a", "004750", "3b586e", "0030d8", "3d008f",
        "ffadfe", "9f1c5d", "9b0010", "d13d00", "ffd951", "7dff00", "00b253",
        "002f35", "003b69", "002bc4", "d5b3ff", "ff53fc", "6f0037", "8f000e",
        "9d2e00", "ffc900", "e3fbcc", "21824f", "00262b", "002b4c", "546bca",
        "ad77ed", "a354a2", "ff3a9b", "ff4151", "6d1f00", "282000", "c2d9ab",
        "006b32", "00dcff", "bde1ff", "0020a2", "8b58cb", "790077", "d22b7b",
        "ff4e5d", "3d1200", "ffec9e", "6dc61a", "69ffb3", "c9f8ff", "007bdf",
        "6370a1", "6e2ebe", "ff7bfb", "532d40", "83000c", "ff7033", "ffe26d",
        "a1b88b", "3cee92", "00c6e2", "4796d7", "001e95", "4b089d", "e893e4",
        "4f0026", "982f37", "ff4c00", "e6d17e", "81986d", "00ff7f", "00b6d5",
        "0066bd", "001982", "7500fc", "c573c3", "c9005f", "3a1215", "ff5c10",
        "483900", "9bff43", "00a955", "00a5c1", "005ca9", "617eff", "9640f5",
        "9b0096", "8a0041", "ff727a", "ffb997", "110e00", "90e843", "002814",
        "0088a0", "4476a1", "c4cfff", "914ee0", "8e008a", "71495c", "ff5e66",
        "ffeee4", "eed149", "637950", "006333", "004450", "59758d", "838ec2",
        "721bd2", "d700cc", "622240", "dd232c", "ffd1b8", "ebd165", "465b34",
        "60eaaa", "51e3ff", "004b88", "00146f", "b46dff", "be34b7", "ff97c5",
        "ff9599", "fa9861", "e0d295", "49a500", "00904d", "00b1d6", "003157",
        "a3aee3", "a958ff", "b200aa", "ffbcda", "bb4f53", "e86d2c", "c7b045",
        "2a3f1a", "00d272", "00a0c1", "a7d6ff", "0022d9", "7400e7", "fc24ee",
        "ffabd1", "582d2e", "431700", "c3b05e", "d3ffb4", "85ffcb", "006b80",
        "0082f4", "3048bd", "3e007d", "e159d8", "f4c5da", "ffb8ba", "ffae81",
        "beb176", "1c4400", "9be0c2", "0c5d6e", "b8d5ef", "00105d", "9a2eff",
        "ff8bf2", "e40068", "df6e70", "ffc4a2", "675400", "b1df94", "7abea1",
        "002c35", "006ed1", "000c4c", "6500c5", "cf07b7", "d2a4b8", "cd3e3f",
        "ffa774", "ffe24f", "4eca00", "00894e", "00d1ff", "97b4cd", "00051c",
        "30005e", "ab0096", "b08498", "774849", "ff8d4a", "ffeb86", "90bd74",
        "004d2b", "8eeaff", "7894ac", "9aa9ff", "cf97ff", "880077", "90325c",
        "ff6e6e", "d57844", "f1d022", "719d56", "bcffe3", "c0f4ff", "00529c",
        "8c9dff", "20003c", "720064", "906679", "976565", "b15a27", "cab024",
        "c0ff9b", "37c78a", "0094b4", "003f76", "788aec", "dbaaff", "650059",
        "7b0038", "140700", "a19140", "527d39", "5b9e82", "005162", "001f3b",
        "001cb6", "cb86ff", "ff9af2", "44001e",
    ]], dtype=np.float
)
DISTINCT_COLORS = DISTINCT_COLORS / 255.0
DISTINCT_COLORS.setflags(write=False)
