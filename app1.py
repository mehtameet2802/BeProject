from flask import Flask, render_template, send_from_directory, url_for
from flask_uploads import UploadSet, IMAGES, configure_uploads
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from typing import Type
from torch import Tensor
from PIL import Image


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ):
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class CNN(nn.Module):
    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int = 1000
    ) -> None:
        super(CNN, self).__init__()
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:

            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion

        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def predMRI(imgPath):
    model = CNN(1, 18, BasicBlock, 2)
    model.load_state_dict(torch.load("cnn100epochclamp01Crop.pt"))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        # transforms.Resize((640, 640)),
        # transforms.CenterCrop((500, 500)),
        transforms.functional.invert,
        # transforms.Resize((224, 224)),
        transforms.CenterCrop((90, 90)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(imgPath)
    img = transform(img)
    maxIntensity = torch.max(img)
    img = torch.clamp(img, 0.4,  maxIntensity - 0.1)
    img = img.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
        print(predictions)
        _, yhat = torch.max(predictions.data, 1)
        p = yhat.item()
    return p


def predFDOPA(imgPath):
    model = CNN(1, 18, BasicBlock, 2)
    model.load_state_dict(torch.load("cnn20epochClamp.pt"))
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((640, 640)),
        transforms.CenterCrop((250, 250)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(imgPath)
    img = transform(img)
    maxIntensity = torch.max(img)
    minIntensity = maxIntensity - 0.175
    mask = (img >= minIntensity) & (img <= maxIntensity)
    img[~mask] = minIntensity - 0.02
    img = img.unsqueeze(0)
    with torch.no_grad():
        predictions = model(img)
        _, yhat = torch.max(predictions.data, 1)
        p = yhat.item()
    return p


app = Flask(__name__, template_folder="templates")


app.config['SECRET_KEY'] = 'aslknfkdnkj'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

file_uri_pref = "D:/College/Be Project/BE/implementation/"

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)


class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)


@app.route('/mri', methods=['GET', 'POST'])
def mri_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        print(f"Result of model = {predMRI(file_uri_pref+file_url)}")
    else:
        file_url = None
    print(f"file url= {file_url}")
    return render_template('index.html',  form=form, file_url=file_url)


@app.route('/fdopa', methods=['GET', 'POST'])
def fdopa_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = url_for('get_file', filename=filename)
        print(f"Result of model = {predFDOPA(file_uri_pref+file_url)}")
    else:
        file_url = None
    print(f"file url= {file_url}")
    return render_template('index.html', form=form, file_url=file_url)


@app.route("/check")
def index():
    return 'Hello, World. Website is working'


@app.route('/')
def hello_world():
    return 'Hello from Flask!'


if __name__ == '__main__':
    app.run(debug=True)
