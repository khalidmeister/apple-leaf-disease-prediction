import io
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, jsonify, request, render_template
from PIL import Image

app = Flask(__name__)

# Modelling Task
model = models.resnet18()
num_inftr = model.fc.in_features
model.fc = nn.Linear(num_inftr, 4)
model.load_state_dict(torch.load('./fix_resnet18.pth'))
model.eval()

class_names = ['Apple Scab', 'Black Rot', 'Cedar Apple Rust', 'Healthy']

def transform_image(image_bytes):
	my_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)


def get_prediction(image_bytes):
	tensor = transform_image(image_bytes=image_bytes)
	outputs = model.forward(tensor)
	_, prediction = torch.max(outputs, 1)
	return class_names[prediction]

diseases = {
	"Healthy": "",
	"Apple Scab": "Apple scab is a common disease of plants in the rose family (Rosaceae) that is caused by the ascomycete fungus Venturia inaequalis. While this disease affects several plant genera, including Sorbus, Cotoneaster, and Pyrus, it is most commonly associated with the infection of Malus trees, including species of flowering crabapple, as well as cultivated apple.",
	"Cedar Apple Rust": "Gymnosporangium juniperi-virginianae is a plant pathogen that causes cedar-apple rust. In virtually any location where apples or crabapples (Malus) and Eastern red-cedar (Juniperus virginiana) coexist, cedar apple rust can be a destructive or disfiguring disease on both the apples and cedars. Quince and hawthorn are the most common host and many species of juniper can substitute for the eastern red cedars.",
	"Black Rot": "Black rot is a fungus disease that can cause serious losses in apple orchards, especially in warm, humid areas. The black rot fungus covers a wide geographical range and can infect many hosts other than apple. The role these hosts play in the spread and development of the disease is not known. Three forms of the disease can occur: a leaf spot known as frogeye leaf spot, a fruit rot, and a limb canker. Severe leaf spotting can result in defoliation that weakens the tree, infected fruit become unmarketable, and limb cankers can girdle and eventually kill entire branches."
}

# Treat the web process
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files.get('file')
		if not file:
			return
		img_bytes = file.read()
		prediction_name = get_prediction(img_bytes)
		return render_template('result.html', name=prediction_name.lower(), description=diseases[prediction_name])

	return render_template('index.html')


if __name__ == '__main__':
	app.run(debug=True)