# TODO:
#   1. Turn large diagonal matrices into sparse matrices (K matrices)
#   2. Implement reduced bloch mode expansion to speed up computation
#   3. Generate a bunch of points of interest other than Gamma, X, and M
#   4. Implement 3D plane wave expansion
#   5. Refactor using object-oriented 'Band' objects, which contain all the points of
#       interest, the bloch vectors, and the eigenfrequencies for each bloch vector.
#   6. Incorporate dispersion (permittivity dependence, and hence convolution matrix dependence, on frequency)

import numpy as np
import scipy as sp
import math as math
from shorthand import *

class Crystal:

    def __init__(self, permittivityCellData, permeabilityCellData, *latticeVectors):
        self.permeabilityCellData = permeabilityCellData
        self.permittivityCellData = permittivityCellData

        self.dimensions = len(latticeVectors)
        self.latticeVectors = latticeVectors
        self.reciprocalLatticeVectors = self.calculateReciprocalLatticeVectors();
        self.crystalType = self.determineCrystalType();
        (self.keySymmetryPoints, self.keySymmetryNames) = self.generateKeySymmetryPoints()
        self.latticeConstant = norm(self.latticeVectors[0]) # TODO: Make this more general

    def calculateReciprocalLatticeVectors(self):
        if self.dimensions is 2:
            return self.calculateReciprocalLatticeVectors2D();
        elif self.dimensions is 3:
            return self.calculateReciprocalLatticeVectors3D();
        else:
            raise NotImplementedError(f"Cannot calculate reciprocal lattice for {self.dimensions}D." +
                    " Not currently implemented.")

    def calculateReciprocalLatticeVectors2D(self):
        rotationMatirx90Degrees = complexArray([
            [0,-1],
            [1,0]]);
        t1 = self.latticeVectors[0];
        t2 = self.latticeVectors[1];

        T1 = 2 * pi * rotationMatirx90Degrees @ t2 / dot(t1, rotationMatirx90Degrees @ t2);
        T2 = 2 * pi * rotationMatirx90Degrees @ t1 / dot(t2, rotationMatirx90Degrees @ t1);
        return (T1, T2);

    def calculateReciprocalLatticeVectors3D(self):
        t1 = self.latticeVectors[0];
        t2 = self.latticeVectors[1];
        t3 = self.latticeVectors[2];
        T1 = 2 * pi * cross(t2, t3) / dot(t1, cross(t2, t3));
        T2 = 2 * pi * cross(t3, t1) / dot(t2, cross(t3, t1));
        T3 = 2 * pi * cross(t1, t2) / dot(t3, cross(t1, t2));

        return (T1, T2, T3);

    def determineCrystalType(self):
        if self.dimensions == 2:
            crystalType = self.determineCrystalType2D()
            return crystalType
        else:
            raise NotImplementedError

    def determineCrystalType2D(self):
        epsilon = 0.00001
        sideLengthDifference = abs(norm(self.reciprocalLatticeVectors[0]) -
                norm(self.reciprocalLatticeVectors[1]))
        latticeVectorProjection = abs(dot(self.reciprocalLatticeVectors[0], self.reciprocalLatticeVectors[1]))

        if sideLengthDifference < epsilon and latticeVectorProjection < epsilon:
            return "SQUARE"
        elif sideLengthDifference > epsilon and latticeVectorProjection < epsilon:
            return "RECTANGULAR"
        elif latticeVectorProjection > epsilon:
            return "OBLIQUE"
        else:
            raise NotImplementedError;

    def generateKeySymmetryPoints(self):
        keySymmetryPoints = []
        keySymmetryNames = []
        T1 = self.reciprocalLatticeVectors[0]
        T2 = self.reciprocalLatticeVectors[1]

        if self.crystalType == "SQUARE":
            keySymmetryNames = ["X", "G", "M"];
            keySymmetryPoints = [0.5 * T1, 0 * T1, 0.5 * (T1 + T2)];
        elif self.crystalType == "RECTANGULAR":
            keySymmetryNames = ["X", "G", "Y", "S"];
            keySymmetryPoints = [0.5 * T1, 0 * T1, 0.5 * T2, 0.5 * (T1 + T2)];
        else:
            raise NotImplementedError;

        return (keySymmetryPoints, keySymmetryNames);

class BandSolver:

    numberHarmonics = None;

    def __init__(self, crystal):
        raise NotImplementedError;

def generateBlochVectors(crystal, internalPointsPerWalk):

    keySymmetryPoints = crystal.keySymmetryPoints
    numberSymmetryPoints = len(keySymmetryPoints);

    blochVectors = [];
    nextSymmetryPoint = None;
    blochVectors.append(keySymmetryPoints[0]);
    for i in range(numberSymmetryPoints - 1):
        currentSymmetryPoint = keySymmetryPoints[i];
        fractionWalked = 1 / (internalPointsPerWalk + 1);

        if(i + 1 < numberSymmetryPoints):
            nextSymmetryPoint = keySymmetryPoints[i + 1];

        for j in range(internalPointsPerWalk + 1):
            deltaVector = nextSymmetryPoint - currentSymmetryPoint;
            desiredPoint = currentSymmetryPoint + (j + 1) * fractionWalked * deltaVector;
            blochVectors.append(desiredPoint);

    return blochVectors;

def calculateZeroHarmonicLocation(*numberHarmonicsTi):
    zeroHarmonicLocations = [];
    for numberHarmonics in numberHarmonicsTi:
        zeroHarmonicLocations.append(math.floor(numberHarmonics / 2));

    return zeroHarmonicLocations;

def calculateMinHarmonic(*numberHarmonicsTi):
    """ Returns the minimum harmonic value (i.e. -2 if there are 5 total harmonics)
    Arguments:
        numberHarmonicsTi: The number of harmonics in the ith direction.
    """
    minHarmonics = [];
    for numberHarmonics in numberHarmonicsTi:
        minHarmonics.append(- math.floor(numberHarmonics / 2));

    return minHarmonics;

def calculateMaxHarmonic(*numberHarmonicsTi):
    """ Returns the maximum harmonic value (i.e. +2 if there are 5 total harmonics, +1 if there are 4)
    Arguments:
        numberHarmonicsTi: The number of harmonics in the ith direction.
    """
    maxHarmonics = [];
    for numberHarmonics in numberHarmonicsTi:
        if(numberHarmonics % 2 == 0):
            maxHarmonics.append(math.floor(numberHarmonics / 2) - 1);
        else:
            maxHarmonics.append(math.floor(numberHarmonics / 2));

    return maxHarmonics;

# This function is too long.
def generateConvolutionMatrix(A, numberHarmonics):
    """
    Generates the 1, 2, or 3D matrix corresponding to the convolution operation with A.
    P: Number of spatial harmonics along x
    Q: Number of spatial harmonics along y
    R: Number of spatial harmonics along z
    """
    # Now, for the code below to work for any dimension we need to add dimensions to A
    dataDimension = len(A.shape);
    (P, Q, R) = numberHarmonics

    convolutionMatrixSize = P*Q*R;
    convolutionMatrixShape = (convolutionMatrixSize, convolutionMatrixSize);
    convolutionMatrix = complexZeros(convolutionMatrixShape)

    A = reshapeLowDimensionalData(A);
    (Nx, Ny, Nz) = A.shape;
    zeroHarmonicXLocation = math.floor(Nx/2);
    zeroHarmonicYLocation = math.floor(Ny/2);
    zeroHarmonicZLocation = math.floor(Nz/2);

    A = fftn(A);

    for rrow in range(R):
        for qrow in range(Q):
            for prow in range(P):
                row = rrow*Q*P + qrow*P + prow;
                for rcol in range(R):
                    for qcol in range(Q):
                        for pcol in range(P):
                            col = rcol*Q*P + qcol*P + pcol;
                            # Get the desired harmonics relative to the 0th-order harmonic.
                            desiredHarmonicZ = rrow - rcol;
                            desiredHarmonicY = qrow - qcol;
                            desiredHarmonicX = prow - pcol;

                            # Get those harmonic locations from the zero harmonic location.
                            desiredHarmonicXLocation = zeroHarmonicXLocation + desiredHarmonicX;
                            desiredHarmonicYLocation = zeroHarmonicYLocation + desiredHarmonicY;
                            desiredHarmonicZLocation = zeroHarmonicZLocation + desiredHarmonicZ;

                            convolutionMatrix[row][col] = \
                                A[desiredHarmonicXLocation][desiredHarmonicYLocation][desiredHarmonicZLocation];
    #print(convolutionMatrix)
    return convolutionMatrix;

def getXComponents(*args):
    """ Gets the x component from a 2 or 3 element row or column vector
    Arguments:
        args: Any number of numpy arrays in row or column vector form
    """
    xComponents = [];
    for a in args:
        if(a.shape == (3,) or a.shape == (2,)): # element is a row vector
            xComponents.append(a[0]);
        elif(a.shape == (3,1) or a.shape == (2,1)): # element is a column vector
            xComponents.append(a[0,0]);

    return xComponents;

def getYComponents(*args):
    """ Gets the y component from a 2 or 3 element row or column vector
    Arguments:
        args: Any number of numpy arrays in row or column vector form
    """
    yComponents = [];

    for a in args:
        if(a.shape == (3,) or a.shape == (2,)):
            yComponents.append(a[1]);
        elif(a.shape == (3,1) or a.shape == (2,1)):
            yComponents.append(a[1,0]);

    return yComponents;


def generateKxMatrix(blochVector, T1, numberHarmonicsT1, T2=complexArray([0,0,0]), numberHarmonicsT2=1,
        T3=complexArray([0,0,0]), numberHarmonicsT3=1):
    """ Generates the Kx matrix for a given bloch vector and number of harmonics
    The matrix this returns assumes a vector that is indexed first over kx, then over ky,
    then over kz. Meaning that the iteration over kz should be in the outermost loop.
    Arguments:
        blochVector: The bloch vector currently under test
        Ti: Reciprocal lattice vector i. Assumed to be a 3-row vector.
        numberHarmonicsTi: Number of harmonics along plane of Ti
    """
    matrixSize = numberHarmonicsT1 * numberHarmonicsT2 * numberHarmonicsT3;
    matrixShape = (matrixSize, matrixSize);
    KxMatrix = complexZeros(matrixShape)

    (blochVectorx, T1x, T2x, T3x) = getXComponents(blochVector, T1, T2, T3);
    (minHarmonicT1, minHarmonicT2, minHarmonicT3) = calculateMinHarmonic(
            numberHarmonicsT1, numberHarmonicsT2, numberHarmonicsT3);
    (maxHarmonicT1, maxHarmonicT2, maxHarmonicT3) = calculateMaxHarmonic(
            numberHarmonicsT1, numberHarmonicsT2, numberHarmonicsT3);

    diagonalIndex = 0;
    for desiredHarmonicT3 in range(minHarmonicT3, maxHarmonicT3 + 1):
        for desiredHarmonicT2 in range(minHarmonicT2, maxHarmonicT2 + 1):
            for desiredHarmonicT1 in range(minHarmonicT1, maxHarmonicT1 + 1):

                KxMatrix[diagonalIndex][diagonalIndex] = blochVectorx - \
                        desiredHarmonicT1*T1x - desiredHarmonicT2*T2x - desiredHarmonicT3*T3x;
                diagonalIndex += 1;

    return KxMatrix;


def generateKyMatrix(blochVector, T1, numberHarmonicsT1, T2=complexArray([0,0,0]), numberHarmonicsT2=1,
        T3=complexArray([0,0,0]), numberHarmonicsT3=1):
    """ Generates the Kx matrix for a given bloch vector and number of harmonics
    The matrix this returns assumes a vector that is indexed first over kx, then over ky,
    then over kz. Meaning that the iteration over kz should be in the outermost loop.
    Arguments:
        blochVector: The bloch vector currently under test
        Ti: Reciprocal lattice vector i. Assumed to be a 3-row vector.
        numberHarmonicsTi: Number of harmonics along plane of Ti
    """
    matrixSize = numberHarmonicsT1 * numberHarmonicsT2 * numberHarmonicsT3;
    matrixShape = (matrixSize, matrixSize);
    KyMatrix = complexZeros(matrixShape)

    (blochVectory, T1y, T2y, T3y) = getYComponents(blochVector, T1, T2, T3);
    (minHarmonicT1, minHarmonicT2, minHarmonicT3) = calculateMinHarmonic(
            numberHarmonicsT1, numberHarmonicsT2, numberHarmonicsT3);
    (maxHarmonicT1, maxHarmonicT2, maxHarmonicT3) = calculateMaxHarmonic(
            numberHarmonicsT1, numberHarmonicsT2, numberHarmonicsT3);

    diagonalIndex = 0;
    for desiredHarmonicT3 in range(minHarmonicT3, maxHarmonicT3 + 1):
        for desiredHarmonicT2 in range(minHarmonicT2, maxHarmonicT2 + 1):
            for desiredHarmonicT1 in range(minHarmonicT1, maxHarmonicT1 + 1):

                KyMatrix[diagonalIndex][diagonalIndex] = blochVectory - \
                        desiredHarmonicT1*T1y - desiredHarmonicT2*T2y - desiredHarmonicT3*T3y;
                diagonalIndex += 1;

    return KyMatrix;

def generateAMatrix(KxMatrix, KyMatrix, erMatrix, urMatrix, mode):
    """ Generates intermediate A matrix from material matrices
    Arguments:
        erMatrix: Convolution matrix for the permittivity tensor
        urMatrix: Convolution matirx for the permeability tensor
        mode: 'E' or 'H'
    """
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

def calculateEigenfrequencies(pointsPerWalk, crystal, numberHarmonics):
    blochVectors = generateBlochVectors(crystal, pointsPerWalk)
    data = [];
    erConvolutionMatrix = generateConvolutionMatrix(crystal.permittivityCellData, numberHarmonics)
    urConvolutionMatrix = generateConvolutionMatrix(crystal.permeabilityCellData, numberHarmonics)
    (T1, T2) = crystal.reciprocalLatticeVectors
    (numberHarmonicsT1, numberHarmonicsT2, x) = numberHarmonics

    # TODO: WE SHOULD CALCULATE BOTH E AND H MODES. CURRENTLY ONLY CALCULATE E MODES.
    for blochVector in blochVectors:
        KxMatrix = generateKxMatrix(blochVector, T1, numberHarmonicsT1,
            T2, numberHarmonicsT2);
        KyMatrix = generateKyMatrix(blochVector, T1, numberHarmonicsT1,
            T2, numberHarmonicsT1);
        AMatrix = generateAMatrix(KxMatrix, KyMatrix, erConvolutionMatrix, urConvolutionMatrix, 'E');
        BMatrix = generateBMatrix(erConvolutionMatrix, urConvolutionMatrix, 'E');
        (DMatrix, VMatrix) = generateVDMatrices(AMatrix, BMatrix);
        eigenValues = np.diagonal(DMatrix);
        eigenFrequencies = scaleEigenvalues(eigenValues, crystal.latticeConstant);
        data.append(eigenFrequencies);

    return (blochVectors, data, crystal.keySymmetryPoints, crystal.keySymmetryNames);

def unwrapBlochVectors(blochVectors):
    """ Unwraps the bloch vectors so that we can plot them on a single axis
    Arguments:
        blochVectors: A list of all bloch vectors you want to unwrap
    """
    xCoordinates = np.array([0]);
    currentCoordinate = 0;
    numPoints = len(blochVectors);

    for i in range(numPoints - 1):
        lastBlochVector = blochVectors[i];
        nextBlochVector = blochVectors[i + 1];
        currentCoordinate += norm(nextBlochVector - lastBlochVector)
        xCoordinates = np.append(xCoordinates, currentCoordinate);

    return xCoordinates;

def generateBandData(blochVectors, eigenFrequencies):
    """ Generates x/y style scatter plot data from a list of bloch vectors and
    eigenfrequencies.
    """
    unwrappedBlochVectors = unwrapBlochVectors(blochVectors);
    numBlochVectors = len(blochVectors);
    numFrequencies = len(eigenFrequencies[0]);

    xData = np.array([]);
    for unwrappedBlochVector in unwrappedBlochVectors:
        xData = np.append(xData, unwrappedBlochVector * np.ones(numFrequencies));
    yData = np.ndarray.flatten(np.array(eigenFrequencies));

    return (xData, yData);

def scaleEigenvalues(eigenValues, a):
    """ Scales the eigenvalues of our system so that they become a normalized frequency """
    return a /(2 * pi) * sqrt(eigenValues);
