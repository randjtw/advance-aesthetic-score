import folder_paths
import clip
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
from PIL import Image

folder_paths.folder_names_and_paths["aesthetic"] = ([os.path.join(folder_paths.models_dir,"aesthetic")], folder_paths.supported_pt_extensions)

class MLP(pl.LightningModule):
	def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
		super().__init__()
		self.input_size = input_size
		self.xcol = xcol
		self.ycol = ycol
		self.layers = nn.Sequential(
			nn.Linear(self.input_size, 1024),
			#nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(1024, 128),
			#nn.ReLU(),
			nn.Dropout(0.2),
			nn.Linear(128, 64),
			#nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(64, 16),
			#nn.ReLU(),
			nn.Linear(16, 1)
		)
	def forward(self, x):
		return self.layers(x)
	def training_step(self, batch, batch_idx):
			x = batch[self.xcol]
			y = batch[self.ycol].reshape(-1, 1)
			x_hat = self.layers(x)
			loss = F.mse_loss(x_hat, y)
			return loss
	def validation_step(self, batch, batch_idx):
		x = batch[self.xcol]
		y = batch[self.ycol].reshape(-1, 1)
		x_hat = self.layers(x)
		loss = F.mse_loss(x_hat, y)
		return loss
	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

class Adv_Scoring:
	def __init__(self):
		pass

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"model_name": (folder_paths.get_filename_list("aesthetic"), {"multiline": False, "default": "chadscorer.pth"}),
				"image": ("IMAGE",),
				"limit": ("FLOAT",{"default": 0.0}),
				}
		}

	RETURN_TYPES = ("NUMBER","FLOAT","STRING","IMAGE")
	FUNCTION = "calc_score"
	CATEGORY = "mynode/scoring"

	def calc_score(self, model_name, image, limit):
		m_path = folder_paths.folder_names_and_paths["aesthetic"][0]
		m_path2 = os.path.join(m_path[0], model_name)
		model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
		s = torch.load(m_path2)
		model.load_state_dict(s)
		model.to("cuda")
		model.eval()
		device = "cuda" 
		model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
		tensor_image = image[0]
		img = (tensor_image * 255).to(torch.uint8).numpy()
		pil_image = Image.fromarray(img, mode='RGB')
		image2 = preprocess(pil_image).unsqueeze(0).to(device)
		with torch.no_grad():
			image_features = model2.encode_image(image2)
		im_emb_arr = normalized(image_features.cpu().detach().numpy())
		prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
		final_prediction = round(float(prediction[0]), 2)
		del model
		if final_prediction < limit:
			image = 1 - image
		return (final_prediction,final_prediction,str(final_prediction),image,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Adv_Scoring": Adv_Scoring
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Adv_Scoring": "Advance Aesthetic Score"
}
