import numpy as np
import matplotlib.pyplot as plt
from skimage import draw
from scipy.ndimage.filters import gaussian_filter
# from skimage.filters import threshold_otsu
#from scipy.ndimage.interpolation import rotate
from skimage.transform import rotate
from PIL import Image
from tqdm import tqdm
from scipy import misc
import imageio
import copy
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("file", help="path to file with image")
parser.add_argument("size", help="size of sampled texture", type=int)
parser.add_argument("num", help="number of sampled textures", type=int)
parser.add_argument("--to_enrich", help="to enrich tiles or not", type=bool, default=False)
parser.add_argument("--n1", help='number of north edges', type=int, default=2)
parser.add_argument("--n2", help='number of west edges', type=int, default=2)
parser.add_argument("--l", help='length of tile + overlap - 1', type=int, default=200)
parser.add_argument("--w", help='width of overlap', type=int, default=80)
args = parser.parse_args()

to_enrich = args.to_enrich
img_path = args.file
TEST_SIZE = args.size
num_to_create = args.num

SIZE = 0
def create_colors_from_texture(texture, n, l, alpha=0):  # only 2 values for alpha 0 or 45
    colors = []
    w = texture.shape[0]
    rotated = rotate(texture, alpha, resize=True)
    #print('rotate', rotated.min())
    #print('rotate', rotated.max())
    for i in tqdm(range(n)):
        if alpha != 0:
            alpha = 45
        alpha = alpha/180*np.pi
        a = w/(np.cos(alpha)+np.sin(alpha))
        l_1 = w*np.cos(alpha) - a*np.cos(alpha)**2
        x = np.random.randint(l_1, rotated.shape[0]-l_1-l) # row
        y = np.random.randint(l_1, rotated.shape[0]-l_1-l) 
    
        img = rotated[x:x+l, y:y+l]
        colors.append(img)
        #print(img.shape)
        #colors.append(img)
        #print('a', img)
    return colors#, t

def find_all_combinations(n1, n2, n_NW):
    comb = np.asarray([str(i)+str(j) for i in range(n1) for j in range(n1, n1+n2)])
    new_comb = []
    used = []
    for k in range(n_NW):
        while True:
            restart = False
            to_choose = np.random.choice(comb, size = len(comb)) # random sample of SE edges
            for u in used:
                if sum(u == to_choose) != 0:
                    to_choose = np.random.choice(comb, size = len(comb))
                    restart = True
                    break
            if not restart:
                break
            used.append(to_choose)
        new_comb = new_comb + [comb[i] + to_choose[i] for i in range(len(comb))]
    return new_comb

def find_all_combinations(n1, n2, n_NW):
    '''
    Finds n_NW combinations of colors
    params:
        n1: number of north colors
        n2: number of west colors
        n_NW: number of north-western color combinations
    returns: list of all color combinations
    '''
    '''
    comb = np.asarray([str(i) + str(j) for i in range(n1) for j in range(n1, n1 + n2)])
    new_comb = []
    used = []
    for k in range(n_NW):
        while True:
            restart = False
            to_choose = np.random.choice(comb, size=len(comb))  # random sample of SE edges
            for u in used:
                if sum(u == to_choose) != 0:
                    to_choose = np.random.choice(comb, size=len(comb))
                    restart = True
                    break
            if not restart:
                break
            used.append(to_choose)
        new_comb = new_comb + [comb[i] + to_choose[i] for i in range(len(comb))]
    '''

    comb = [str(i) + str(j) for i in range(n1) for j in range(n1, n1 + n2)]
    comb = [x+y for x in comb for y in comb]
    return comb


# correct!!!! (almost)
def merge_two_parts(arr1, arr2, omega, axis=1): # omega is width of overlap
    h,w = arr1.shape
    prev = np.empty((arr1.shape[0], omega), dtype=list)
    distances = np.full((arr1.shape[0], omega), fill_value=np.inf)
    
    distances[0] = np.ones(omega)
    
    if axis == 0:
        arr1 = arr1.swapaxes(1,0)
        arr2 = arr2.swapaxes(1,0)
        #merged = np.hstack([arr2, arr1[:,omega:]]) # resulting image

    merged = np.hstack([arr1, arr2[:,omega:]]) 

    for i in range(1, arr1.shape[0]):
        for j in range(omega):
            cur = (arr1[i,w-omega+j]-arr2[i,j])**2
            weight_cur = distances[i-1,j]
                
            if j < omega-1 and j > 0:
                weight_prev = distances[i-1,j-1]
                weight_next = distances[i-1,j+1]
                
                if weight_cur <= weight_prev:
                    if weight_cur <= weight_next:
                        distances[i,j] = distances[i-1,j]+cur
                        prev[i,j] = (i-1,j)
                    else:
                        distances[i,j] = distances[i-1,j+1]+cur
                        prev[i,j] = (i-1,j+1)
                else:
                    if weight_prev <= weight_next:
                        distances[i,j] = distances[i-1,j-1]+cur
                        prev[i,j]= (i-1,j-1)
                    else:
                        distances[i,j] = distances[i-1,j+1]+cur
                        prev[i,j] = (i-1,j+1)
                        
            elif j < omega-1:
                weight_next = distances[i-1,j+1]
                if weight_cur <= weight_next:
                    distances[i,j] = distances[i-1,j]+cur
                    prev[i,j] = (i-1,j)
                else:
                    distances[i,j] = distances[i-1,j+1]+cur
                    prev[i,j] = (i-1,j+1)
                    
            elif j > 0:
                weight_prev = distances[i-1,j-1]
        
                if weight_cur <= weight_prev:
                    distances[i,j] = distances[i-1,j]+cur
                    prev[i,j] = (i-1,j)
                else:
                    distances[i,j] = distances[i-1,j-1]+cur
                    prev[i,j] = (i-1,j-1)
            else:
                distances[i,j] = distances[i-1,j]+cur
                prev[i,j] = (i-1,j)
    
    j = np.argmin(distances[h-1])
    
    for i in range(h):

        merged[h-1-i,w-omega+j:w] = arr2[h-1-i,j:omega]
        if i != h-1:
            i_prev, j_prev = prev[h-1-i,j]
            j = j_prev
    if axis == 0:
        return merged[:,:w].swapaxes(1,0), merged[:,w-omega:].swapaxes(1,0)
    else:
        return merged[:,:w], merged[:,w-omega:]
    
    
def create_tilings(colors_comb, colors, n1, n2, n_total, a, l, omega):
    n_rep = int(n_total/n1**2/n2**2)
    h,w = colors[0].shape
    size = l-omega//2
    tilings = {}
    center = l-1-omega//2 # l - size of image crop, a - resulting size of tile
    for i in range(n_total):
        c = []
        for num in colors_comb[i]:
            #print(num)
            c.append(colors[int(num)][:])
        final = np.zeros((2*size, 2*size))
        
       
        for k in range(4):
           
            if k == 0:
                c[k], c[(k+1)%4] = merge_two_parts(c[k], c[(k+1)%4], omega)
                merged = np.hstack([c[k][:,:w-omega], c[(k+1)%4]])
                final[:l] = merged

            elif k == 1:
                c[1] = np.rot90(c[1], k=1)
                c[2] = np.rot90(c[2], k=1)
                c[1], c[2] = merge_two_parts(c[1], c[2], omega)
                merged = np.hstack([c[1][:,:-omega], c[2]])
                final[:,-l:] = np.rot90(merged, k=1, axes=(1, 0))
                c[1] = np.rot90(c[1], k=1, axes=(1, 0))
                c[2] = np.rot90(c[2], k=1, axes=(1, 0))
            elif k == 2:                
                c[3], c[2] = merge_two_parts(c[3], c[2], omega)
                merged = np.hstack([c[3], c[2][:,omega:]])
                final[-l:] = merged
            elif k == 3:
                c[0] = np.rot90(c[0], k=1)
                c[3] = np.rot90(c[3], k=1)
                c[0], c[3] = merge_two_parts(c[0], c[3], omega)
                merged = np.hstack([c[0][:,:-omega], c[3]])
                final[:,:l] = np.rot90(merged, k=1, axes=(1,0))
                c[0] = np.rot90(c[0], k=1, axes=(1, 0))
                c[3] = np.rot90(c[3], k=1, axes=(1, 0))
                

            '''
            for j in range(omega//2, l):
                if k == 0:
                    tile_ind = j-omega//2
                    if tile_ind < a:
                        ind1 = tile_ind
                        indices2 = np.arange(a-tile_ind-1,a)
                        indices1 = np.arange(ind1+1) 
                    else:
                        ind1 = 2*a-1-tile_ind-1
                        indices2 = np.arange(ind1+1)
                        indices1 = np.arange(tile_ind-a+1, a)
                    final[indices1, indices2] = merged[j, np.arange(center-ind1, center+ind1+1,2)]
                elif k == 1:
                    tile_ind = j-omega//2
                    if tile_ind < a:
                        ind1 = tile_ind
                        indices1 = np.arange(a-tile_ind-1,a)
                        indices2 = np.arange(a-1, a-tile_ind-1-1,-1)
                    else:
                        ind1 = 2*a - 1 - tile_ind
                        indices1 = np.arange(ind1+1)
                        indices2 = np.arange(2*a-1-tile_ind, -1, -1)

                    final[indices1, indices2] = merged[j, np.arange(center-ind1, center+ind1+1,2)]
                elif k == 2:
                    tile_ind = j-omega//2
                    if tile_ind < a:
                        ind1 = tile_ind
                        indices1 = np.arange(a-1,a-1-tile_ind-1,-1)
                        indices2 = np.arange(tile_ind,-1,-1)
                    else:
                        ind1 = 2*a - 1 - tile_ind
                        indices1 = np.arange(ind1, -1, -1)
                        indices2 = np.arange(a-1, (a-ind1-2), -1)

                    final[indices1, indices2] = merged[j, np.arange(center-ind1, center+ind1+1,2)]
                elif k == 3:
                    tile_ind = j-omega//2
                    if tile_ind < a:
                        ind1 = tile_ind
                        indices1 = np.arange(tile_ind, -1, -1)
                        indices2 = np.arange(tile_ind+1)
                    else:
                        ind1 = 2*a - 1 - tile_ind
                        indices1 = np.arange(a-1, tile_ind-a-1, -1)
                        indices2 = np.arange(tile_ind-a, a)
                    final[indices1, indices2] = merged[j, np.arange(center-ind1, center+ind1+1,2)]
            
            if i == 0:
                tmp = c[2][:]
            '''
            #final = np.rot90(final, k = k, axes=(1, 0))

            plt.imsave('final_{0}_{1}.png'.format(k,colors_comb[i]), final, cmap='gray')
        #print(final.shape)
        final = rotate(final[omega//2:-omega//2, omega//2:-omega//2], -45, resize=True)
        #print(final.shape)
        #print(size)
        a = final.shape[0]//2
        print(a)
        center = final.shape[0] - a
      
        plt.imsave('final_{}.png'.format(colors_comb[i]), final[a//2:-a//2, a//2:-a//2], cmap='gray')
        #print(final[a//2:-a//2, a//2:-a//2].shape)
        #print(size)
        #print(colors_comb[i])
        
        tilings.update({colors_comb[i]: final[a//2:-a//2, a//2:-a//2]})
        #print(colors_comb[i])
    #print(tilings.keys())
    return a, tilings


def enrich_tiles(tiles, enrichments, width):
    enriched_tiles = copy.deepcopy(tiles)
    center_tile = tiles[list(tiles.keys())[0]].shape[0] // 2
    center_enrichment = enrichments[0].shape[0] // 2
    delta = center_tile - center_enrichment
    #print(delta)
    i = 0
    for ind, tile in tiles.items():
        ar1, ar2 = merge_two_parts(np.rot90(tile[:width + delta, delta:-delta]),
                                   np.rot90(enrichments[i][:width]), width)

        enrichments[i][:width] = np.rot90(ar2, axes=(1, 0))

        ar1, ar2 = merge_two_parts(enrichments[i][:, -width:], tile[delta:-delta, -(width + delta):], width)
        enrichments[i][:, -width:] = ar1

        ar1, ar2 = merge_two_parts(np.rot90(enrichments[i][-width:]),
                                   np.rot90(tile[-(width + delta):, delta:-delta]), width)
        enrichments[i][-width:] = np.rot90(ar1, axes=(1, 0))

        ar1, ar2 = merge_two_parts(tile[delta:-delta, :(width + delta)], enrichments[i][:, :width], width)
        enrichments[i][:, :width] = ar2
        enriched_tiles[ind][delta:-delta, delta:-delta] = enrichments[i]
        i += 1
    return enriched_tiles


def create_tiles(image, n_1, n_2, n_NW, l, w, patch_border, width_enrich, enrich=False):
    n_cs = n_1 * n_2
    n_total = int(n_NW * n_cs)
    res_size = l - w - 1

    colors = create_colors_from_texture(image, n_1 + n_2, l, 45)
    patch_size = res_size - patch_border
    comb = find_all_combinations(n_1, n_2, n_NW)
    res_size, tiles = create_tilings(comb[:n_total], colors, n_1, n_2, n_total, res_size, l, w)

    if enrich:
        enrichments = create_colors_from_texture(image, n_total, patch_size, 0)
        enriched_tiles = enrich_tiles(tiles, enrichments, width_enrich)
        # tiles_1 = copy.deepcopy(tiles)
        tiles = enriched_tiles

    return tiles, comb, res_size


def create_tiled_img_from_source(image, n_1,n_2, n_NW, l, w, patch_border, width_enrich, size, enrich=False):
    n_cs = n_1*n_2
    n_total = int(n_NW*n_cs)
    a = SIZE
    
    colors = create_colors_from_texture(image, n_1+n_2, l, 45)
    comb = find_all_combinations(n_1, n_2, n_NW)
    #print(comb)
    #print(n_total)
    a, tiles = create_tilings(comb[:n_total], colors, n_1, n_2, n_total, a, l, w)
    #print(SIZE)
    #a = SIZE
    height = size//a+1
    width = size//a+1

    
    patch_size = a - patch_border
    
    if enrich:
        #print(tiles[comb[0]].shape)
        #print(enrichments[0].shape)
        enrichments = create_colors_from_texture(image, n_total, patch_size, 0)
        enriched_tiles = enrich_tiles(tiles, enrichments, width_enrich)
        tiles_1 = copy.deepcopy(tiles)
        tiles = enriched_tiles
    
    prev_border = random.sample(tiles.keys(), 1)[0]
    prev_tile = tiles[prev_border]
    #print(prev_tile.shape)
    a = prev_tile.shape[0]
    whole_img = np.zeros((height*a, width*a))


    whole_img[:a,:a] = prev_tile
    #print(whole_img.shape)
    prev_line = ['' for i in range(width)]
    prev_line[0] = prev_border
    #print(prev_border)
    for i in range(height):
        for j in range(width):
            if i+j == 0:
                continue
            #print(tiles.keys())
            if i == 0:
                #print(j)
                result = {key: value for key, value in tiles.items() if key[3] == prev_border[1]}
            elif j == 0:
                #print(i,j)
                result = {key: value for key, value in tiles.items() if key[0] == prev_line[j][2]}
            else:
                #print(i,j)
                #print(prev_border)
                #print(prev_line[j])
                result = {key: value for key, value in tiles.items() \
                          if key[3] == prev_border[1] and key[0] == prev_line[j][2]}
            
            prev_border = random.sample(result.keys(), 1)[0]
            #print(prev_border)
            prev_tile = result[prev_border]
            prev_line[j] = prev_border
            whole_img[i*a:(i+1)*a, j*a:(j+1)*a] = prev_tile
    return whole_img[:size, :size]


if __name__ == '__main__':
    img = imageio.imread(img_path)
    w = img.shape[0]
    #print(img.max())
    if len(img.shape) != 2:
        img = img[:, :w, 0].astype(float) / 255.
    else:
        img = img[:, :w].astype(float) / 255.
    print(img.max())
    #n_1 = 2
    #n_2 = 2
    #n_NW = n_1 * n_2

    #print(img.min())
    #print(img.max())
    n_1 = args.n1
    n_2 = args.n2

    n_NW = n_1 * n_2

    l = args.l
    w = args.w
    patch_border = 6
    file_name, ext = img_path.split('/')[-1].split('.')

    #tiles, combinations, a = create_tiles(img, n_1, n_2, n_NW, l, w, patch_border=80, width_enrich=30, enrich=to_enrich)
    for i in range(num_to_create):
        tiled = create_tiled_img_from_source(img, n_1, n_2, n_NW, l, w, 80, 30, TEST_SIZE, enrich=False)
        #print(tiled.min())
        #print(tiled.max())
        plt.imsave('berea_'+str(i+1)+'.jpg', tiled, cmap='gray')
        Image.open('wang_alporas/alporas_'+str(i+1)+'.jpg').convert('RGB').save('wang_alporas/alporas_'+str(i+1)+'.jpg')
        #print(tiled.max())
        #print(tiled.min())
        #Image.fromarray((tiled*255).astype(np.uint8)).save('wang_res/'+file_name+'_'+str(i+1)+'.'+ext)
