import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.metrics import structural_similarity as ssim

folder_path = '/Users/luryand/Documents/DPI/dip-2024-2/img'
output_dir = 'output_segmentations'

specific_files = [
    "flowers.jpg", "gecko.png", "rice.tif", "beans.png",
    "blobs.png", "chips.png", "coffee.png", "dowels.tif"
]

def load_images(folder_path, file_list):
    images = {}
    for filename in file_list:
        img_path = os.path.join(folder_path, filename)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        images[filename] = img
    return images

def canny_edges_density(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    edges = cv.Canny(blurred, 100, 200)
    edge_density = (np.sum(edges == 255) / edges.size) * 100
    return edges, edge_density

def otsu_mask_threshold(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    otsu_threshold, mask = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    fg_percentage = (np.sum(mask == 255) / mask.size) * 100
    return mask, otsu_threshold, fg_percentage

def watershed_segmentation(image):
    blurred = cv.GaussianBlur(image, (5, 5), 0)
    thresh, mask_otsu = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(mask_otsu, cv.MORPH_OPEN, kernel, iterations = 2)
    sure_bg = cv.dilate(opening, kernel, iterations=3)
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    thresh_dist, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)
    ret, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown==255] = 0
    img_color = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    markers = cv.watershed(img_color, markers)
    labels = np.unique(markers)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, len(labels)))
    np.random.shuffle(colors)
    colors[0] = [0, 0, 0, 1]
    segmented_colored = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label == -1:
             segmented_colored[markers == label] = [255, 255, 255]
        elif label != 0 :
            segmented_colored[markers == label] = (colors[i][:3] * 255).astype(np.uint8)
    return segmented_colored

def calc_similarity(img1, img2):
    if img1.shape != img2.shape:
        target_shape = (img1.shape[1], img1.shape[0])
        img2_resized = cv.resize(img2, target_shape, interpolation=cv.INTER_NEAREST)
    else:
        img2_resized = img2
    dr = 255.0
    ssim_value = ssim(img1, img2_resized, data_range=dr)
    mse_value = np.mean((img1.astype(np.float64) - img2_resized.astype(np.float64)) ** 2)
    if np.isinf(mse_value): mse_value = float('inf')
    return {'ssim': ssim_value, 'mse': mse_value}


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    loaded_images = load_images(folder_path, specific_files)
    filenames = list(loaded_images.keys())
    num_images = len(filenames)

    otsu_masks_dict = {}
    segmentation_results_for_saving = {}

    print("\n------ Análise Individual -------")
    for filename, img in loaded_images.items():
        edges_canny, density = canny_edges_density(img)
        print(f"  {filename:<12} [Descontinuidade]: Densidade Bordas={density:.2f}%")

        mask_otsu, threshold, fg_perc = otsu_mask_threshold(img)
        otsu_masks_dict[filename] = mask_otsu
        print(f"  {filename:<12} [Similaridade]   : Limiar Otsu={threshold:<6.1f}, Foreground={fg_perc:.1f}%")

        segmented_watershed = watershed_segmentation(img)
        segmentation_results_for_saving[filename] = {
            'original': img,
            'canny': edges_canny,
            'otsu': mask_otsu,
            'watershed': segmented_watershed
        }

    for filename, results in segmentation_results_for_saving.items():
        fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
        ax = axes.ravel()

        ax[0].imshow(results['original'], cmap='gray')
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].imshow(results['canny'], cmap='gray')
        ax[1].set_title('Canny Edges')
        ax[1].axis('off')

        ax[2].imshow(results['otsu'], cmap='gray')
        ax[2].set_title('Otsu Mask')
        ax[2].axis('off')

        ax[3].imshow(results['watershed'])
        ax[3].set_title('Watershed')
        ax[3].axis('off')

        fig.suptitle(f'Segmentações - {filename}')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_path = os.path.join(output_dir, f"segmentation_{os.path.splitext(filename)[0]}.png")
        plt.savefig(output_path)
        plt.close(fig)

    print("\n--- Comparação das Máscaras geradas no Otsu (SSIM / MSE) ---")
    if num_images >= 2:
        print(f"{'':<14}" + "".join([f"{name[:6]:<10}" for name in filenames]))
        print("-" * (14 + 10 * num_images))

        for i in range(num_images):
            fname1 = filenames[i]
            print(f"{fname1[:12]:<12} |", end="")
            for j in range(num_images):
                fname2 = filenames[j]
                if i == j:
                    print(f"{'---':<10}", end="")
                elif j < i:
                     print(f"{'':<10}", end="")
                else:
                    mask1 = otsu_masks_dict[fname1]
                    mask2 = otsu_masks_dict[fname2]
                    similarity = calc_similarity(mask1, mask2)
                    ssim_str = f"{similarity['ssim']:.2f}"
                    mse_val = similarity['mse']
                    mse_str = "Inf" if mse_val == float('inf') else f"{mse_val:.0f}"
                    print(f"{ssim_str}/{mse_str}"[:9].ljust(9) + "|", end="")
            print()

    print("\nUsei métodos vistos em aula e que estavam no slide, foram: Detecção por Descontinuidade (Canny) e Similaridade (Otsu e Watershed). Em resumo da aplicação, apliquei Gaussian Blur para redução de ruído, depois processei com as chamadas operações morfológicas para o Watershed receber. Pra cada imagem plotaram as bordas de Canny, a máscara binária Otsu e a segmentação colorida por regiões do Watershed, adicionei também a imagem original pra comparar na hora.")
    print("\nÉ interessante ver como o watershed consegue segmentar objetos com diferentes cores, imagino que tenha uma boa utilidade pra reconhecer diferentes objetos na mesma imagem, segmentar geometrias que tem um certo contato entre elas mas reconhecer ambas como diferentes objetos. Achei interessante isso!!")
    print("\nImplementei tudo com OpenCV como perguntado em aula e usei o sk-image pra fazer a parte de métricas, tudo da documentação.")
    
    print(f"\nSalvei outputs em: ./output_segmentations")