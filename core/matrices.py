# TODO:
#   1. Turn large diagonal matrices into sparse matrices (K matrices)
#   2. Implement reduced bloch mode expansion to speed up computation
#   3. Generate a bunch of points of interest other than Gamma, X, and M
#   4. Implement 3D plane wave expansion

import numpy as np
import scipy as sp
import scipy.linalg
import math as math

inv = np.linalg.inv;
matrixExponentiate = sp.linalg.expm
matrixSquareRoot = sp.linalg.sqrtm
sqrt = np.lib.scimath.sqrt; # Takes sqrt of complex numbers
sq = np.square;
eig = sp.linalg.eig # Performs eigendecomposition of identity intuitively (vectors are unit vectors)
norm = np.linalg.norm;
sin = np.sin;
cos = np.cos;
pi = np.pi;

OUTER_BLOCK_SHAPE = (2,2);
scatteringElementShape = (2,2); # The shape of our core PQ matrices.
scatteringMatrixShape = OUTER_BLOCK_SHAPE + scatteringElementShape;
scatteringElementShape = (2,2);
scatteringElementSize = scatteringElementShape[0];
DBGLVL = 2;

def fftn(data):
    """ Return the shifted version so the zeroth-order harmonic is in the center with
    energy-conserving normalization """
    dataShape = data.shape;
    return np.fft.fftshift(np.fft.fftn(data)) / np.prod(dataShape);

def complexArray(arrayInListForm):
    """ Wrapper for numpy array declaration that forces arrays to be complex doubles """
    return np.array(arrayInListForm, dtype=np.cdouble);

def complexIdentity(matrixSize):
    """ Wrapper for numpy identity declaration that forces arrays to be complex doubles """
    return np.identity(matrixSize, dtype=np.cdouble);

def complexZeros(matrixDimensionsTuple):
    """ Wrapper for numpy zeros declaration that forces arrays to be complex doubles """
    return np.zeros(matrixDimensionsTuple, dtype=np.cdouble);

def complexOnes(matrixDimensionsTuple):
    return np.ones(matrixDimensionsTuple, dtype=np.cdouble);

def reshapeLowDimensionalData(data):
    dataShape = data.shape;
    if(len(dataShape) == 1): # we have only x-data.
        Nx = dataShape[0];
        data = data.reshape(Nx, 1, 1);
    elif(len(dataShape) == 2): # We have x and y data
            Nx = dataShape[0];
            Ny = dataShape[1];
            data = data.reshape(Nx, Ny, 1);
    elif(len(dataShape) == 3): # We have x- y- and z-data (
        data = data;
    else:
        raise ValueError(f"""Input data has too many ({len(dataShape)}) dimensions.
        Only designed for up to 3 spatial dimensions""");

    return data;

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
def generateConvolutionMatrix(A, P, Q=1, R=1):
    """
    Generates the 1, 2, or 3D matrix corresponding to the convolution operation with A.
    P: Number of spatial harmonics along x
    Q: Number of spatial harmonics along y
    R: Number of spatial harmonics along z
    """
    # Now, for the code below to work for any dimension we need to add dimensions to A
    dataDimension = len(A.shape);

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
    """ Generates intermediate B matrix from material matrices
    Arguments:
        erMatrix: Convolution matrix for the permittivity tensor
        urMatrix: Convolution matirx for the permeability tensor
        mode: 'E' or 'H'
    """
    materialMatrix = None;
    if(mode == 'E'):
        materialMatrix = erMatrix;
    elif(mode == 'H'):
        materialMatrix = urMatrix;
    else:
        raise Exception(f"Undefined mode {mode}. Choose E or H");

    return materialMatrix;

def generateVDMatrices(AMatrix, BMatrix):
    """ Computes and sorts the eigenvalues and eigenvectors
    Arguments:
        AMatrix: The generalized eigenvalue problem operator including the curls and stuff
        BMatrix: The material convolution matrix that we don't want to invert
    """
    # THIS CURRENTLY COMPUTES THE CORRECT EIGENVALUES BUT NOT THE RIGHT EIGENVECTORS.
    eigenValues, eigenVectors = eig(AMatrix, BMatrix);
    indices = np.flip(eigenValues.argsort()[::-1]);
    eigenValues = eigenValues[indices];
    eigenVectors = eigenVectors[indices, :];

    eigenValueMatrix = np.diag(eigenValues);
    #eigenVectorMatrix = np.transpose(eigenVectors);
    #print("\nEigenvalue [0]:");
    #print(f"{eigenValues[0]}");
    #print("\nEigenvectors[0]:");
    #print(f"{eigenVectors[0]}");
    #print(f"Norm:{sp.linalg.norm(eigenVectors[0])}")

    return (eigenValueMatrix, eigenVectors);

def calculateEigenfrequencies(pointsPerWalk, er, ur, t1, numberHarmonicsT1, t2=complexArray([0,0,0]),
        numberHarmonicsT2=1, t3=complexArray([0,0,0]), numberHarmonicsT3=1):
    """ Computes from a given Bloch vector and lattice vectors the eigenfrequencies of that system
    Arguments:
        pointsPerWalk: The number of points between two points of symmetry in the desired band diagram
    """
    # ALSO, WE NEED TO COMPUTE A. FOR NOW, WE WILL JUST SAY THIS IS THE LENGTH OF THE REAL SPACE LATTICE VECTOR.
    a = norm(t1);

    # First, compute the reciprocal lattice vectors from the real-space lattice vectors.
    # NEED TO BUILD THESE FUNCTIONS.
    (T1, T2, T3) = calculateReciprocalLatticeVectors(t1, t2, t3);
    blochVectors = generateBlochVectors(pointsPerWalk, symmetryType, T1, T2, T3);
    data = [];

    # Now, generate the convolution matrices which we will use for the rest of our program.
    erConvolutionMatrix = generateConvolutionMatrix(er, numberHarmonicsT1, numberHarmonicsT2, numberHarmonicsT3);
    urConvolutionMatrix = generateConvolutionMatrix(ur, numberHarmonicsT1, numberHarmonicsT2, numberHarmonicsT3);

    # Finally, loop through each bloch vector to calculate all the possible eigenfrequencies for that vector.
    # ALSO: WE SHOULD CALCULATE BOTH E AND H MODES 
    for blochVector in blochVectors:
        KxMatrix = generateKxMatrix(blochVector, T1, numberHarmonicsT1,
            T2, numberHarmonicsT2, T3, numberHarmonicsT3);
        KyMatrix = generateKyMatrix(blochVector, T1, numberHarmonicsT1,
            T2, numberHarmonicsT1, T3, numberHarmonicsT3);
        AMatrix = generateAMatrix(KxMatrix, KyMatrix, erConvolutionMatrix, urConvolutionMatrix, 'E');
        BMatrix = generateBMatrix(KxMatrix, KyMatrix, erConvolutionMatrix, urConvolutionMatrix, 'E');
        (DMatrix, VMatrix) = generateVDMatrices(AMatrix, BMatrix);
        eigenValues = np.diagonal(DMatrix);
        eigenFrequenicies = scaleEigenvalues(eigenValues, a);
        data.append(eigenFrequencies);

    return (blochVectors, data);


def scaleEigenvalues(eigenValues, a):
    """ Scales the eigenvalues of our system so that they become a normalized frequency """
    return a /(2 * pi) * sqrt(eigenValues);
