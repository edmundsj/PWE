# TODO:
#   1. Turn large diagonal matrices into sparse matrices (K matrices)
#   2. Implement reduced bloch mode expansion to speed up computation
#   4. Implement 3D plane wave expansion
#   6. Incorporate dispersion (permittivity dependence, and hence convolution matrix dependence, on frequency)
#   7. Plot only N lowest number of bands (so as not to clutter the plot)
#   8. Annotate the plot with the points of special symmetry
#   9. Implement convergence testing using an increasing number of harmonics
#       Also, auto-choose the number of harmonics based on the power contained in the Fourier spetrum of
#       the specified permittivity and permeability fourier transform matrices
#   10. Add netlist parser for standard shapes rather than doing it directly in python/ allow other programs
#       to generate the data
#   11. This is only valid for square and rectangular symmetries because the Bloch wave expansion is only
#       valid in that case. (Lecture 19 CEM 8:33). Figure out how to generalize this.

import sys
import numpy as np
import scipy as sp
import math as math
import matplotlib.pyplot as plt

from shorthand import *
from convolution import *
from harmonics import *
from crystal import *

class BandStructure:

    def __init__(self, crystal):
        self.crystal = crystal
        self.blochVectors = []
        self.unwrappedBlochVectors = np.array([])
        self.frequencyData = []

    def Solve(self, numberHarmonics, numberInternalPoints):
        self.generateBlochVectors(numberInternalPoints)

        erConvolutionMatrix = generateConvolutionMatrix(self.crystal.permittivityCellData, numberHarmonics)
        urConvolutionMatrix = generateConvolutionMatrix(self.crystal.permeabilityCellData, numberHarmonics)
        (numberHarmonicsT1, numberHarmonicsT2, x) = numberHarmonics

        # TODO: WE SHOULD CALCULATE BOTH E AND H MODES. CURRENTLY ONLY CALCULATE E MODES.
        for blochVector in self.blochVectors:
            KxMatrix = generateKxMatrix(blochVector, self.crystal, numberHarmonics)
            KyMatrix = generateKyMatrix(blochVector, self.crystal, numberHarmonics)
            AMatrix = generateAMatrix(KxMatrix, KyMatrix, erConvolutionMatrix, urConvolutionMatrix, 'E');
            BMatrix = generateBMatrix(erConvolutionMatrix, urConvolutionMatrix, 'E');
            (DMatrix, VMatrix) = generateVDMatrices(AMatrix, BMatrix);
            eigenValues = np.diagonal(DMatrix);
            eigenFrequencies = self.scaleEigenvalues(eigenValues);
            self.frequencyData.append(eigenFrequencies);

        self.generateBandData()

    def Plot(self, wmax=2):
        plt.scatter(self.xScatterPlotData, self.yScatterPlotData)
        plt.ylim(0, wmax)
        plt.show()

    def generateBlochVectors(self, internalPointsPerWalk):

        numberSymmetryPoints = len(self.crystal.keySymmetryPoints);

        self.blochVectors = [];
        nextSymmetryPoint = None;
        self.blochVectors.append(self.crystal.keySymmetryPoints[0]);

        for i in range(numberSymmetryPoints - 1):
            currentSymmetryPoint = self.crystal.keySymmetryPoints[i];
            fractionWalked = 1 / (internalPointsPerWalk + 1);

            if(i + 1 < numberSymmetryPoints):
                nextSymmetryPoint = self.crystal.keySymmetryPoints[i + 1];

            for j in range(internalPointsPerWalk + 1):
                deltaVector = nextSymmetryPoint - currentSymmetryPoint;
                desiredPoint = currentSymmetryPoint + (j + 1) * fractionWalked * deltaVector;
                self.blochVectors.append(desiredPoint);

    def unwrapBlochVectors(self):
        """ Unwraps the bloch vectors so that we can plot them on a single axis
        Arguments:
            blochVectors: A list of all bloch vectors you want to unwrap
        """
        self.unwrappedBlochVectors = np.array([0]);
        currentCoordinate = 0;
        numPoints = len(self.blochVectors);

        for i in range(numPoints - 1):
            lastBlochVector = self.blochVectors[i];
            nextBlochVector = self.blochVectors[i + 1];
            currentCoordinate += norm(nextBlochVector - lastBlochVector)
            self.unwrappedBlochVectors = np.append(self.unwrappedBlochVectors, currentCoordinate);

    def generateBandData(self):
        self.unwrapBlochVectors()
        numBlochVectors = len(self.blochVectors);
        numFrequencies = len(self.frequencyData[0]);

        self.xScatterPlotData = np.array([]);
        for unwrappedBlochVector in self.unwrappedBlochVectors:
            self.xScatterPlotData = np.append(self.xScatterPlotData,
                    unwrappedBlochVector * np.ones(numFrequencies));
        yData = np.ndarray.flatten(np.array(self.frequencyData));

        self.yScatterPlotData = yData

    def scaleEigenvalues(self, eigenValues):
        """ Scales the eigenvalues of our system so that they become a normalized frequency """
        return self.crystal.latticeConstant /(2 * pi) * sqrt(eigenValues);

def generateAMatrix(KxMatrix, KyMatrix, erMatrix, urMatrix, mode):
    if(mode == 'E'):
        urMatrixInverse = inv(urMatrix);
        AMatrix = KxMatrix @ urMatrixInverse @ KxMatrix + KyMatrix @ urMatrixInverse @ KyMatrix;
    elif(mode == 'H'):
        erMatrixInverse = inv(erMatrix);
        AMatrix = KxMatrix @ erMatrixInverse @ KxMatrix + KyMatrix @ erMatrixInverse @ KyMatrix;
    else:
        raise Exception(f"Undefined mode {mode}. Choose E or H");

    return AMatrix;

def generateBMatrix(erMatrix, urMatrix, mode):
    materialMatrix = None;
    if(mode == 'E'):
        materialMatrix = erMatrix;
    elif(mode == 'H'):
        materialMatrix = urMatrix;
    else:
        raise Exception(f"Undefined mode {mode}. Choose E or H");

    return materialMatrix;

def generateVDMatrices(AMatrix, BMatrix):
    # THIS CURRENTLY COMPUTES THE CORRECT EIGENVALUES BUT NOT THE RIGHT EIGENVECTORS AND I DON'T KNOW WHY.
    eigenValues, eigenVectors = eig(AMatrix, BMatrix);
    indices = np.flip(eigenValues.argsort()[::-1]);
    eigenValues = eigenValues[indices];
    eigenVectors = eigenVectors[indices, :];

    eigenValueMatrix = np.diag(eigenValues);

    return (eigenValueMatrix, eigenVectors);

