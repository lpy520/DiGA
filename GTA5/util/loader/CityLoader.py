import os.path as osp
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
from PIL import Image, ImageFile
from .augmentations import *
ImageFile.LOAD_TRUNCATED_IMAGES = True

valid_colors = [[0,  0, 0],
                [255,  0, 0],
                [ 0,  255,  0],
                [255, 255, 0]]
label_colours = dict(zip(range(4), valid_colors))

class CityLoader(data.Dataset):
	def __init__(self, root, img_list_path, lbl_list_path, max_iters=None, crop_size=None, mean=(128, 128, 128), transform=None, pseudo_dir = None, set='val', return_name = False, use_pseudo = False):
		self.n_classes = 19
		self.root = root
		self.crop_size = crop_size
		self.mean = mean
		self.transform = transform
		self.return_name = return_name
		self.use_pseudo = use_pseudo
		self.pseudo_dir = pseudo_dir
		# self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
		self.img_ids = [i_id.strip() for i_id in open(img_list_path)]
		self.lbl_ids = [i_id.strip() for i_id in open(lbl_list_path)]
		#if self.use_pseudo:
			#self.pseudo_lbl_ids = [i_id.strip() for i_id in open(img_list_path)]

		if not max_iters==None:
		   self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
		   self.lbl_ids = self.lbl_ids * int(np.ceil(float(max_iters) / len(self.lbl_ids)))
		   #if self.use_pseudo:
			   #self.pseudo_lbl_ids = self.pseudo_lbl_ids * int(np.ceil(float(max_iters) / len(self.pseudo_lbl_ids)))


		self.files = []
		self.id_to_trainid = {0: 0, 1 : 1, 2: 2, 3: 3}
		self.set = set
		# for split in ["train", "trainval", "val"]:
		for img_name, lbl_name in zip(self.img_ids, self.lbl_ids):
			img_file = osp.join(self.root, "images/%s" % (img_name))
			lbl_file = osp.join(self.root, "labels/%s" % (lbl_name))
			if self.use_pseudo:
				pseudo_name = img_name.split('/')[-1]
				pseudo_lbl_file = osp.join(self.root, "%s/%s" % (self.pseudo_dir, pseudo_name))
				self.files.append({
					"img": img_file,
					"label": lbl_file,
					"pseudo_label": pseudo_lbl_file,
					"name": img_name
				})
			else:
				self.files.append({
					"img": img_file,
					"label": lbl_file,
					"name": img_name
				})

	def __len__(self):
		return len(self.files)

	def __getitem__(self, index):
		datafiles = self.files[index]

		image = Image.open(datafiles["img"]).convert('RGB')
		label = Image.open(datafiles["label"])
		if self.use_pseudo:
			pseudo_label = Image.open(datafiles["pseudo_label"])
		name = datafiles["name"]

		# resize
		if self.crop_size != None:
			image = image.resize((self.crop_size[1], self.crop_size[0]), Image.BICUBIC)
			label = label.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)
			if self.use_pseudo:
				pseudo_label = pseudo_label.resize((self.crop_size[1], self.crop_size[0]), Image.NEAREST)

		# transform
		if self.transform != None:
			if self.use_pseudo:
				image, label, pseudo_label = self.transform(image, label, pseudo_label)
			else:
				image, label = self.transform(image, label)

		image = np.asarray(image, np.float32)
		image = image[:, :, ::-1]  # change to BGR
		image -= self.mean
		image = image.transpose((2, 0, 1)) / 128.0

		if not self.use_pseudo:
			label = np.asarray(label, np.compat.long)

			# re-assign labels to match the format of Cityscapes
			label_copy = 255 * np.ones(label.shape, dtype=np.compat.long)
			for k, v in self.id_to_trainid.items():
				label_copy[label == k] = v

			if not self.return_name:
				return image.copy(), label_copy.copy()
			else:
				return image.copy(), label_copy.copy(), name
		else:
			label = np.asarray(label, np.compat.long)
			pseudo_label = np.asarray(pseudo_label, np.compat.long)

			# re-assign labels to match the format of Cityscapes
			label_copy = 255 * np.ones(label.shape, dtype=np.compat.long)
			for k, v in self.id_to_trainid.items():
				label_copy[label == k] = v

			pseudo_label_copy = 255 * np.ones(pseudo_label.shape, dtype=np.compat.long)
			for v in range(self.n_classes):
				pseudo_label_copy[pseudo_label == v] = v


			if not self.return_name:
				return image.copy(), label_copy.copy(), pseudo_label_copy.copy()
			else:
				return image.copy(), label_copy.copy(), pseudo_label_copy.copy(), name

	def decode_segmap(self, img):
		map = np.zeros((img.shape[0], img.shape[1], img.shape[2], 3))
		for idx in range(img.shape[0]):
			temp = img[idx, :, :]
			r = temp.copy()
			g = temp.copy()
			b = temp.copy()
			for l in range(0, self.n_classes):
				r[temp == l] = label_colours[l][0]
				g[temp == l] = label_colours[l][1]
				b[temp == l] = label_colours[l][2]
	
			rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
			rgb[:, :, 0] = r / 255.0
			rgb[:, :, 1] = g / 255.0
			rgb[:, :, 2] = b / 255.0
			map[idx, :, :, :] = rgb
		return map

if __name__ == '__main__':
	dst = GTA5DataSet("./data", is_transform=True)
	trainloader = data.DataLoader(dst, batch_size=4)
	for i, data in enumerate(trainloader):
		imgs, labels = data
		if i == 0:
			img = torchvision.utils.make_grid(imgs).numpy()
			img = np.transpose(img, (1, 2, 0))
			img = img[:, :, ::-1]
			plt.imshow(img)
			plt.show()
