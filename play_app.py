import sys
from PyQt5 import uic, QtWidgets, QtCore
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import numpy.linalg as linalg
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pandas import ExcelWriter
from mayavi.api import Engine
from mayavi.sources.api import ParametricSurface
from mayavi.modules.api import Surface
from mayavi import mlab
from tvtk.tools import visual

from PIL import Image
qtCreatorFile = "Cellscanner.ui" 

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class Imagen(object):
    def __init__(self):
        self.img = None
        self.img_procesada = None
        self.img_3deizada = None
        self.contornos = []
        self.centroX = None
        self.centroY = []
        self.ancho = []
        self.largo = []
        self.angulos = []

    # SET
    def set_img(self, img_inp):
        self.img = img_inp

    def set_img_procesada(self, img_inp):
        self.img_procesada = img_inp

    def set_img_3deizada(self, img_inp):
        self.img_3deizada = img_inp

    def set_contornos(self, contornos_inp):
        self.contornos = contornos_inp

    def set_centroX(self, centrosX_inp):
        self.centrosX = centrosX_inp

    def set_centroY(self, centrosY_inp):
        self.centroY = centrosY_inp

    def set_ancho(self, anchos_inp):
        self.ancho = anchos_inp

    def set_largo(self, largos_inp):
        self.largo = largos_inp

    def set_angulos(self, angulos_inp):
        self.angulo = angulos_inp

    # GET
    def get_img(self):
        return self.img

    def get_img_procesada(self):
        return self.img_procesada

    def get_img_3deizada(self):
        return self.img_3deizada

    def get_contornos(self):
        return self.contornos

    def get_centroX(self):
        return self.centrosX

    def get_centroY(self):
        return self.centroY

    def get_ancho(self):
        return self.ancho

    def get_largo(self):
        return self.largo

    def get_angulos(self):
        return self.angulo

    def procesamiento(self, frame):
        kernel = np.ones((5,5),np.uint8)
        kernel5 = np.ones((2,2),np.uint8)
        kernel6 = np.ones((4,4),np.uint8)
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imagen2=cv2.bitwise_not(gris)
        _, thresh= cv2.threshold(imagen2,149,255,cv2.THRESH_BINARY)
        _, thresh2= cv2.threshold(imagen2,220,255,cv2.THRESH_BINARY)

        thresh=cv2.bitwise_not(thresh)

        DE = cv2.erode(thresh,kernel, iterations=1)
        EF = cv2.dilate(DE,kernel6,iterations=1)
        FG = cv2.erode(thresh2,kernel5,iterations=1)

        or4=cv2.bitwise_or(EF,FG)
        not4=cv2.bitwise_not(or4)
        return not4

    def contornear(self, imagen_canny, ancho_max, ancho_min, largo_max, largo_min):
        (img, contornos, jerarquias) = cv2.findContours(imagen_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        X = []
        Y = []
        Ma = []
        mA = []
        Angle = []
        contornos2 = []

        for i in range(0, len(contornos)):#se guardan los valores de las elipses que mas se asemejan a las bacterias
            if(len(contornos[i])>4):
                (x,y),(MA,ma),angle = cv2.fitEllipse(contornos[i])
                #x, y son el centro, MA es el ancho y ma el largo
                if ((MA < ancho_max) and (MA > ancho_min) and (ma <largo_max) and (ma > largo_min)):
                    contornos2.append(contornos[i])
                    ellipse = cv2.fitEllipse(contornos[i])
                    X.append(x)
                    Y.append(y)
                    Ma.append(MA)
                    mA.append(ma)
                    Angle.append(angle)
        return contornos2, X, Y, Ma, mA, Angle

    def tresdeizar(self, X,Y,anchox, anchoy, angulo): #Crea una aproximación a 3D en base a los contornos encontrados en el 2D, adaptando las formas a elipses.
        engine = Engine()
        engine.start()
        scene = engine.new_scene()
        scene.scene.disable_render = True
        print("tresdeizando")
        visual.set_viewer(scene)

        surfaces = []
        for k in range(0,len(X)):
            source = ParametricSurface()
            source.function = 'ellipsoid'
            engine.add_source(source)

            surface = Surface()
            source.add_module(surface)
            
            actor = surface.actor 

            actor.property.opacity = 0.7
            actor.property.color = (0,0,1)
            actor.mapper.scalar_visibility = False 

            actor.actor.orientation = np.array([90,angulo[k],0])

            actor.actor.position = np.array([X[k],Y[k],0])
            actor.actor.scale = np.array([anchox[k]/2, anchox[k]/2, anchoy[k]/2] )

            surfaces.append(surface)

            source.scene.background = (1.0,1.0,1.0)
        print("tresdeizado completado, mostrando :D")
        CellScann.set_img_3deizada(mlab)
        return mlab.show()

    def procesar(self):
        imagen = CellScann.get_img()
        cannyzada = self.procesamiento(imagen)

        contornos,  X, Y, Ma, mA, Angle = self.contornear(cannyzada, 10, 4, 35, 5)
        CellScann.set_contornos(np.array(contornos, dtype=object))
        CellScann.set_centroX(np.array(X))
        CellScann.set_centroY(np.array(Y))
        CellScann.set_ancho(np.array(Ma))
        CellScann.set_largo(np.array(mA))
        CellScann.set_angulos(np.array(Angle))
        
        for i in range(0, len(X)):#se dibujan las elipses con bordes amplificados de todas las elipses
            ellipse = (X[i], Y[i]), (Ma[i], mA[i]), Angle[i]
            ima = cv2.ellipse(imagen,ellipse,(0,0,255),1, cv2.LINE_AA)

        ima_pil = Image.fromarray(ima)

        procesada = cv2.cvtColor(ima, cv2.COLOR_BGR2RGB)
        CellScann.set_img_procesada(procesada)
        return procesada
        
        
    def tresdeizacion(self): 
        self.tresdeizar(CellScann.get_centroX(), CellScann.get_centroY(), CellScann.get_ancho(), CellScann.get_largo(), CellScann.get_angulos())

    def videizacion(self):
        # Pendiente
        pass


CellScann = Imagen()


class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.Procesar.clicked.connect(self.procesar)
        self.Tresdeizar.clicked.connect(self.tresdeizar)
        self.Videizacion.clicked.connect(self.videizacion)

        self.Imagen = QtWidgets.QLabel(" ")
        self.Layout.addWidget(self.Imagen)
        self.Abrir_imagen.clicked.connect(self.getfile)

        self.datos = QtWidgets.QLabel(" ")
        self.Layout2.addWidget(self.datos)
        self.Layout2.setAlignment(QtCore.Qt.AlignLeft) 

        self.titulo_datos = QtWidgets.QLabel("Datos:")
        self.Datos.addWidget(self.titulo_datos)
        self.Datos.setAlignment(QtCore.Qt.AlignLeft) 

    def getfile(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', '~/', "Image files (*.jpg *.gif *.jpeg)")
        if fname:
            with open(fname[0], "rb") as file:
                data = np.array(bytearray(file.read()))

                image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
                CellScann.set_img(image)
                self.mostrarImagen(image)

    def mostrarImagen(self, image):
        size = image.shape
        step = image.size / size[0]
        qformat = QImage.Format_Indexed8

        if len(size) == 3:
            if size[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(image, size[1], size[0], step, qformat)
        img = img.rgbSwapped()

        self.Imagen.setPixmap(QPixmap.fromImage(img))
        self.resize(self.Imagen.pixmap().size())

    def procesar(self):
        ima = CellScann.procesar()
        
        self.mostrarImagen(CellScann.get_img_procesada())
        decimales = 4
        self.datos.setText("""Promedio anchos: {} \nPromedio largos: {} \nDesviacion anchos: {} \nDesviacion largos: {} \nMediana anchos: {} \nMediana largos: {} \nVarianza anchos: {} \nVarianza largos: {} \n
		""".format(str(round(np.mean(CellScann.get_ancho()),decimales)), str(round(np.mean(CellScann.get_largo()),decimales)), str(round(np.std(CellScann.get_ancho()),decimales)), str(round(np.std(CellScann.get_largo()),decimales)), str(round(np.median(CellScann.get_ancho()),decimales)), str(round(np.median(CellScann.get_largo()),decimales)), str(round(np.var(CellScann.get_ancho()),decimales)), str(round(np.var(CellScann.get_largo()),decimales))))

    def tresdeizar(self):
        CellScann.tresdeizacion()

    def videizacion(self):
        pass


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
