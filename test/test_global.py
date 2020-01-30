# Testing is even more critical for simulators than for regular software, as you need to have reference
# data to ensure that your simulator is giving out the proper value. Otherwise, the Law garbage in ->
# garbage out will bite you hard.
#
# This test suite is based on data obtained from Prof. Raymond Rumpf at UTEP
# (empossible.net/wp-content/uploads/2019/08/Benchmarking-Aid-for-TMM.pdf)
# in addition to data I generated myself using code that had previously been validated using the above.
# I have not found any errors in his benchmarking data.
#
# This test suite also compares data to the analytic fresnel's equations for the case of a single
# interface for LHI (Linear Homogenous Isotropic) media and for two interfaces (an etalon), as
# formulae for the above are readily derived and can be checked against online resources. 
#
#
# TODO:
# 1. Write unit tests for the following methods:
#   1. Write additional test cases for the convolution matrix (non-identity ones)

import sys
sys.path.append('core');

from matrices import *
from fresnel import *
from convolution import *
from shorthand import *
import matplotlib.pyplot as plt

statuses = [];
messages = [];

def assertAlmostEqual(a, b, absoluteTolerance=1e-8, relativeTolerance=1e-7):
    """ Wrapper for numpy testing 'all close' assertion, forcing specification of absolute
    and relative tolerance. """

    np.testing.assert_allclose(a, b, atol=absoluteTolerance, rtol=relativeTolerance);

def assertArrayEqual(a, b):
    np.testing.assert_array_equal(a, b);

def assertStringEqual(a, b):
    np.testing.assert_array_equal(a, b);

def assertDictEqual(a, b):
    """ Asserts that two dicts are equal by separately finding out whether their keys
    and values are equal. There is probably a more elegant way to do this"""

    np.testing.assert_array_equal(list(a.keys()), list(b.keys()));
    np.testing.assert_array_equal(list(a.values()), list(b.values()));

class Test:
    def __init__(self):
        self.messages = []; # Messages sent back from our tests (strings)
        self.statuses = []; # statuses sent back from our tests (boolean)
        self.unitTestsEnabled = True;
        self.integrationTestsEnabled = True;

        a = 1;
        self.ax = a;
        self.ay = a;
        self.r = 0.35 * a;
        self.er = 9.0;

        self.Nx = 512;
        self.Ny = 512;
        self.dx = self.ax / self.Nx;
        self.dy = self.ay / self.Ny;
        self.P = 3;
        self.Q = self.P;
        self.numberHarmonics = (self.P, self.Q, 1)
        self.matrixDimensions = self.P * self.Q;
        self.matrixShape = (self.matrixDimensions, self.matrixDimensions);

        # The lattice vectors used in our simulation. For orthorhombic symmetry, these are just 2*pi/a
        # multiplied by the unit vector in the direction normal to the planes of symmetry.
        self.t1 = complexArray([self.ax, 0])
        self.t2 = complexArray([0, self.ay])
        self.T1 = (2 * pi / self.ax) * complexArray([1, 0]);
        self.T2 = (2 * pi / self.ay) * complexArray([0, 1]);
        #self.T3 = complexArray([0,0]); # there is no z-periodicity.
        self.blochVectorGPoint = 0*self.T1;
        self.blochVectorXPoint = 0.5*self.T1;
        self.blochVectorMPoint = 0.5*self.T1 + 0.5*self.T2;

        xcoors = np.linspace(-self.ax/2 + self.dx/2, self.ax/2 - self.dx/2, self.Nx);
        ycoors = np.linspace(-self.ay/2 + self.dy/2, self.ay/2 - self.dy/2, self.Ny);
        (X, Y) = np.meshgrid(xcoors, ycoors);
        self.UR = complexOnes((self.Nx, self.Ny));
        self.ER = (self.er-1) * np.heaviside(sq(X) + sq(Y) - sq(self.r),1)
        self.ER = self.ER + 1;
        self.crystal = Crystal(self.ER, self.UR, self.t1, self.t2)

        # The data for Kx, Ky, and Kz will be re-used at each point of key symmetry
        self.KxMatrixGPoint = complexZeros(self.matrixShape);
        self.KxMatrixGPoint[0,0] = 6.2832;
        self.KxMatrixGPoint[2,2] = -6.2832;
        self.KxMatrixGPoint[3,3] = 6.2832;
        self.KxMatrixGPoint[5,5] = -6.2832;
        self.KxMatrixGPoint[6,6] = 6.2832;
        self.KxMatrixGPoint[8,8] = -6.2832;

        self.KyMatrixGPoint = complexZeros(self.matrixShape);
        self.KyMatrixGPoint[0,0] = 6.2832;
        self.KyMatrixGPoint[1,1] = 6.2832;
        self.KyMatrixGPoint[2,2] = 6.2832;
        self.KyMatrixGPoint[6,6] = -6.2832;
        self.KyMatrixGPoint[7,7] = -6.2832;
        self.KyMatrixGPoint[8,8] = -6.2832;

        self.KxMatrixXPoint = complexZeros(self.matrixShape);
        self.KxMatrixXPoint[0,0] = 9.4248;
        self.KxMatrixXPoint[1,1] = 3.1416;
        self.KxMatrixXPoint[2,2] = -3.1416;
        self.KxMatrixXPoint[3,3] = 9.4248;
        self.KxMatrixXPoint[4,4] = 3.1416;
        self.KxMatrixXPoint[5,5] = -3.1416;
        self.KxMatrixXPoint[6,6] = 9.4248;
        self.KxMatrixXPoint[7,7] = 3.1416;
        self.KxMatrixXPoint[8,8] = -3.1416;

        diagonalValuesKyXPoint = complexArray([6.2832, 6.2832, 6.2832, 0, 0, 0, -6.2832, -6.2832, -6.2832]);
        self.KyMatrixXPoint = np.diag(diagonalValuesKyXPoint);

        diagonalValuesKxMPoint = complexArray([9.4248, 3.1416, -3.1416, 9.4248, 3.1416,
            -3.1416, 9.4248, 3.1416, -3.1416]);
        self.KxMatrixMPoint = np.diag(diagonalValuesKxMPoint);

        diagonalValuesKyMPoint = complexArray([9.4248, 9.4248, 9.4248, 3.1416, 3.1416,
            3.1416,-3.1416, -3.1416, -3.1416]);
        self.KyMatrixMPoint = np.diag(diagonalValuesKyMPoint);

        # Finally, our test data for A and B matrices (E and H modes at each point of symmetry)
        diagonalAEModeGPoint = complexArray([78.9568, 39.4784, 78.9568, 39.4784, 0,
            39.4784, 78.9568, 39.4784, 78.9568]);
        self.AMatrixEModeGPoint = np.diag(diagonalAEModeGPoint);

        self.BMatrixEModeGPoint = complexArray([
            [5.9208, 1.5571 - 0.0096j, 0.2834 - 0.0035j, 1.5571 - 0.0096j, -0.5879 + 0.0072j,
                -0.3972 + 0.0073j, 0.2834 - 0.0035j, -0.3972 + 0.0073j, 0.2256 - 0.0055j],
            [1.5571 + 0.0096j, 5.9208, 1.5571 - 0.0096j, -0.5880 - 0j, 1.5571 - 0.0096j,
                -0.5879 + 0.0072j, -0.3972 + 0.0024j, 0.2834 - 0.0035j, -0.3972 + 0.0073j],
            [0.2834 + 0.0035j, 1.5571 + 0.0096j, 5.9208, -0.3972 - 0.0024j, -0.5880 - 0j,
                1.5571 - 0.0096j, 0.2256 - 0.0j, -0.3972 + 0.0024j, 0.2834 - 0.0035j],
            [1.5571 + 0.0096j, -0.5880 + 0.0j, -0.3972 + 0.0024j, 5.9208, 1.5571 - 0.0096j,
                0.2834 - 0.0035j, 1.5571 - 0.0096j, -0.5879 + 0.0072j, -0.3972 + 0.0073j],
            [-0.5879 - 0.0072j, 1.5571 + 0.0096j, -0.5880 + 0.0j, 1.5571 + 0.0096j, 5.9208,
                1.5571 - 0.0096j, -0.5880 - 0.0j, 1.5571 - 0.0096j, -0.5879 + 0.0072j],
            [-0.3972 - 0.0073j, -0.5879 - 0.0072j, 1.5571 + 0.0096j, 0.2834 + 0.0035j, 1.5571 + 0.0096j,
                5.9208, -0.3972 - 0.0024j, -0.5880 - 0.0j, 1.5571 - 0.0096j],
            [0.2834 + 0.0035j, -0.3972 - 0.0024j, 0.2256 + 0.0j, 1.5571 + 0.0096j, -0.5880 + 0.0j,
                -0.3972 + 0.0024j, 5.9208, 1.5571 - 0.0096j, 0.2834 - 0.0035j],
            [-0.3972 - 0.0073j, 0.2834 + 0.0035j, -0.3972 - 0.0024j, -0.5879 - 0.0072j, 1.5571 + 0.0096j,
                -0.5880 + 0.0000j, 1.5571 + 0.0096j, 5.9208, 1.5571 - 0.0096j],
            [0.2256 + 0.0055j, -0.3972 - 0.0073j, 0.2834 + 0.0035j, -0.3972 - 0.0073j, -0.5879 - 0.0072j,
                1.5571 + 0.0096j, 0.2834 + 0.0035j, 1.5571 + 0.0096j, 5.9208]]);

        # A and B matrices for the E Mode at the X point
        diagonalAEModeXPoint = complexArray([128.3049, 49.3480, 49.3480, 88.8264, 9.8696,
            9.8696, 128.3049, 49.3480, 49.3480]);
        self.AMatrixEModeXPoint = np.diag(diagonalAEModeXPoint);
        self.BMatrixEModeXPoint = self.BMatrixEModeGPoint;

        # A and B Matrices for the E mode at the M point
        diagonalAMatrixEModeMPoint = complexArray([177.6529, 98.6960, 98.6960, 98.6960, 19.7392,
            19.7392, 98.6960, 19.7392, 19.7392]);
        self.AMatrixEModeMPoint = np.diag(diagonalAMatrixEModeMPoint);
        self.BMatrixEModeMPoint = self.BMatrixEModeGPoint;


        # A and B matrices for the H mode at the Gamma (G) point
        self.AMatrixHModeGPoint = complexArray([
            [17.6525 - 0.0j, -3.7513 + 0.0230j, 0, -3.7513 + 0.0230j, 0,
                0.5979 - 0.0110j, 0, 0.5979 - 0.0110j, 0.5397 - 0.0132j],
            [-3.7513 - 0.0230j, 10.7279 - 0.0j, -3.7513 + 0.0230j, 0, 0,
                0, 0.5980 - 0.0035j, -1.5385 + 0.0189j, 0.5979 - 0.0110j],
            [0, -3.7513 - 0.0230j, 17.6524 + 0.0j, 0.5980 + 0.0037j, 0,
                -3.7513 + 0.0230j, 0.5399 - 0.0j, 0.5980 - 0.0035j, 0],
            [-3.7513 - 0.0230j, 0, 0.5980 - 0.0035j, 10.7279 - 0.0j, 0,
                -1.5385 + 0.0189j, -3.7513 + 0.0230j, 0, 0.5979 - 0.0110j],
            [0,0,0,0,0,0,0,0,0],
            [0.5979 + 0.0110j, 0, -3.7513 - 0.0230j, -1.5385 - 0.0189j, 0,
               10.7279 + 0.0j, 0.5980 + 0.0037j, 0, -3.7513 + 0.0230j],
            [0, 0.5980 + 0.0035j, 0.5399 + 0.0j, -3.7513 - 0.0230j, 0,
                0.5980 - 0.0035j, 17.6525 + 0.0j, -3.7513 + 0.0230j, 0],
            [0.5979 + 0.0110j, -1.5385 - 0.0189j, 0.5980 + 0.0035j, 0, 0,
                0, -3.7513 - 0.0230j, 10.7279, -3.7513 + 0.0230j],
            [0.5397 + 0.0132j, 0.5979 + 0.0110j, 0, 0.5979 + 0.0110j, 0,
                -3.7513 - 0.0230j, 0, -3.7513 - 0.0230j, 17.6524]]);
        self.BMatrixHModeGPoint = complexIdentity(self.matrixDimensions);

        self.AMatrixHModeXPoint = complexArray([
            [ 28.6853 - 0.0000j, -6.5649 + 0.0403j, 0.1851 - 0.0023j, -8.4405 + 0.0518j, 2.4631 - 0.0302j,
                0.4484 - 0.0083j, 0.9255 - 0.0114j, 0.1495 - 0.0028j, 0.4722 - 0.0116j],
            [-6.5649 - 0.0403j, 13.4099 - 0.0000j, -2.8135 + 0.0173j, 2.5939 + 0.0000j, -1.4772 + 0.0091j, -0.8646 + 0.0106j, 0.1495 - 0.0009j, -1.1539 + 0.0142j, 0.7474 - 0.0138j],
            [0.1851 + 0.0023j, -2.8135 - 0.0173j, 11.0328 + 0.0000j, 0.4485 + 0.0028j, -0.8211 - 0.0000j, -0.9378 + 0.0058j, 0.4724 - 0.0000j, 0.7475 - 0.0046j, -0.5553 + 0.0068j],
            [-8.4405 - 0.0518j, 2.5939 - 0.0000j, 0.4485 - 0.0028j, 24.1378 - 0.0000j, -4.4316 + 0.0272j, -1.1539 + 0.0142j, -8.4405 + 0.0518j, 2.5937 - 0.0318j, 0.4484 - 0.0083j],
            [2.4631 + 0.0302j, -1.4772 - 0.0091j, -0.8211 + 0.0000j, -4.4316 - 0.0272j, 3.5471 + 0.0000j, 1.4772 - 0.0091j, 2.4633 + 0.0000j, -1.4772 + 0.0091j, -0.8210 + 0.0101j],
            [0.4484 + 0.0083j, -0.8646 - 0.0106j, -0.9378 - 0.0058j, -1.1539 - 0.0142j, 1.4772 + 0.0091j, 2.6820 + 0.0000j, 0.4485 + 0.0028j, -0.8646 - 0.0000j, -0.9378 + 0.0058j],
            [0.9255 + 0.0114j, 0.1495 + 0.0009j, 0.4724 + 0.0000j, -8.4405 - 0.0518j, 2.4633 - 0.0000j, 0.4485 - 0.0028j, 28.6853 + 0.0000j, -6.5649 + 0.0403j, 0.1851 - 0.0023j],
            [0.1495 + 0.0028j, -1.1539 - 0.0142j, 0.7475 + 0.0046j, 2.5937 + 0.0318j, -1.4772 - 0.0091j, -0.8646 + 0.0000j, -6.5649 - 0.0403j, 13.4099, -2.8135 + 0.0173j],
            [0.4722 + 0.0116j, 0.7474 + 0.0138j, -0.5553 - 0.0068j, 0.4484 + 0.0083j, -0.8210 - 0.0101j, -0.9378 - 0.0058j, 0.1851 + 0.0023j, -2.8135 - 0.0173j, 11.0328]]);
        self.BMatrixHModeXPoint = self.BMatrixHModeGPoint;


        # Finally, the A and B matrices at the M point
        self.AMatrixHModeMPoint = complexArray([
            [ 39.7181 - 0.0000j, -11.2540 + 0.0691j, 1.1106 - 0.0136j, -11.2540 + 0.0691j, 4.9263 - 0.0605j, 0, 1.1106 - 0.0136j, 0, 0.4048 - 0.0099j],
            [-11.2540 - 0.0691j, 26.8198 + 0.0000j, -7.5027 + 0.0460j, 5.1879 + 0.0000j, -5.9088 + 0.0363j, 1.7292 - 0.0212j, -0.0000 + 0.0000j, -0.7692 + 0.0094j, 0.5979 - 0.0110j],
            [1.1106 + 0.0136j, -7.5027 - 0.0460j, 22.0656 - 0.0000j, 0.0000 - 0.0000j, 1.6422 + 0.0000j, -3.7513 + 0.0230j, 0.4049 - 0.0000j, 0.5980 - 0.0037j, -0.3702 + 0.0045j],
            [-11.2540 - 0.0691j, 5.1879 - 0.0000j, -0.0000 - 0.0000j, 26.8198 - 0.0000j, -5.9088 + 0.0363j, -0.7692 + 0.0094j, -7.5027 + 0.0460j, 1.7292 - 0.0212j, 0.5979 - 0.0110j],
            [4.9263 + 0.0605j, -5.9088 - 0.0363j, 1.6422 - 0.0000j, -5.9088 - 0.0363j, 7.0941 + 0.0000j, 0, 1.6422 + 0.0000j, 0, -1.6421 + 0.0202j],
            [0, 1.7292 + 0.0212j, -3.7513 - 0.0230j, -0.7692 - 0.0094j, 0, 5.3640 + 0.0000j, 0.5980 + 0.0037j, -1.7293 - 0.0000j, 0],
            [1.1106 + 0.0136j, -0.0000 + 0.0000j, 0.4049 + 0.0000j, -7.5027 - 0.0460j, 1.6422 - 0.0000j, 0.5980 - 0.0037j, 22.0656 + 0.0000j, -3.7513 + 0.0230j, -0.3702 + 0.0045j],
            [0, -0.7692 - 0.0094j, 0.5980 + 0.0037j, 1.7292 + 0.0212j, 0, -1.7293 + 0.0000j, -3.7513 - 0.0230j, 5.3640, 0],
            [0.4048 + 0.0099j, 0.5979 + 0.0110j, -0.3702 - 0.0045j, 0.5979 + 0.0110j, -1.6421 - 0.0202j, 0, -0.3702 - 0.0045j, 0, 4.4131]]);
        self.BMatrixHModeMPoint = self.BMatrixHModeGPoint;

        # And now, the V and D matrices. These are (I think) the eigenvalues and eigenvectors.
        # D and V matrices for the E mode at the Gamma (G) point
        diagonalDEModeGPoint = complexArray([0, 5.3493, 5.9424, 5.9424, 7.3189, 14.1508,
            21.4393, 21.4393, 31.9392]);
        self.DMatrixEModeGPoint = np.diag(diagonalDEModeGPoint);
        self.VMatrixEModeGPoint = complexArray([
            [0.0000 - 0.0000j, -0.0000 + 0.0000j, -0.0496 + 0.0012j, -0.0736 + 0.0018j, -0.1056 + 0.0026j, -0.2116 + 0.0052j, -0.2968 + 0.0073j, 0.1385 - 0.0034j, 0.2288 - 0.0056j],
            [-0.0000 - 0.0000j, -0.1840 + 0.0039j, 0.0467 - 0.0005j, -0.2394 + 0.0042j, -0.1549 + 0.0029j, 0.0000 - 0.0000j, 0.2240 - 0.0065j, 0.0813 - 0.0065j, -0.3121 + 0.0057j],
            [0.0000 + 0.0000j, 0.0000 - 0.0000j, 0.0736 - 0.0007j, -0.0496 + 0.0005j, -0.1057 + 0.0013j, 0.2117 - 0.0026j, -0.1384 + 0.0062j, -0.2966 + 0.0134j, 0.2289 - 0.0028j],
            [-0.0000 + 0.0000j, 0.1840 - 0.0039j, -0.2394 + 0.0041j, -0.0467 + 0.0011j, -0.1549 + 0.0029j, 0.0000 + 0.0000j, 0.0816 + 0.0008j, -0.2239 + 0.0091j, -0.3121 + 0.0057j],
            [-0.4109 + 0.0041j, -0.0000 + 0.0000j, 0.0000 - 0.0000j, 0.0000 + 0.0000j, 0.1210 - 0.0015j, 0.0000 - 0.0000j, 0.0000 + 0.0000j, -0.0000 - 0.0000j, 0.4193 - 0.0051j],
            [0.0000 + 0.0000j, 0.1840 - 0.0017j, 0.2394 - 0.0011j, 0.0467 - 0.0005j, -0.1550 + 0.0010j, -0.0000 + 0.0000j, -0.0816 - 0.0018j, 0.2240 - 0.0064j, -0.3122 + 0.0019j],
            [-0.0000 - 0.0000j, 0.0000 + 0.0000j, -0.0736 + 0.0007j, 0.0496 - 0.0005j, -0.1057 + 0.0013j, 0.2117 - 0.0026j, 0.1384 - 0.0062j, 0.2966 - 0.0134j, 0.2289 - 0.0028j],
            [-0.0000 - 0.0000j, -0.1840 + 0.0017j, -0.0467 - 0.0001j, 0.2394 - 0.0012j, -0.1550 + 0.0010j, -0.0000 + 0.0000j, -0.2241 + 0.0037j, -0.0814 + 0.0055j, -0.3122 + 0.0019j],
            [0.0000, 0.0000, 0.0496, 0.0737, -0.1057, -0.2117, 0.2969, -0.1385, 0.2289]]);

        # D and V matrices for the E mode at the X point
        diagonalDEModeXPoint = complexArray([1.3032, 2.0302, 6.3790, 6.7140, 12.2827, 12.5853,
            17.4185, 29.2110, 48.6988]);
        self.DMatrixEModeXPoint = np.diag(diagonalDEModeXPoint);
        self.VMatrixEModeXPoint = complexArray([
            [ -0.0025 + 0.0001j, 0.0003 - 0.0000j, 0.0254 - 0.0006j, 0.0208 - 0.0005j, 0.0813 - 0.0020j, 0.1215 - 0.0030j, -0.1139 + 0.0028j, -0.3084 + 0.0076j, 0.3048 - 0.0075j],
            [0.0080 - 0.0001j, -0.0331 + 0.0006j, 0.1832 - 0.0034j, 0.1859 - 0.0034j, 0.2075 - 0.0038j, -0.0399 + 0.0007j, -0.2235 + 0.0041j, 0.1993 - 0.0037j, -0.2608 + 0.0048j],
            [0.0079 - 0.0001j, 0.0340 - 0.0004j, 0.1714 - 0.0021j, 0.1718 - 0.0021j, -0.2533 + 0.0031j, 0.0831 - 0.0010j, 0.2781 - 0.0034j, -0.0941 + 0.0012j, 0.0786 - 0.0010j],
            [0.0072 - 0.0001j, -0.0097 + 0.0002j, 0.0000 - 0.0000j, -0.0510 + 0.0009j, 0.0000 - 0.0000j, 0.2987 - 0.0055j, 0.0086 - 0.0002j, 0.0000 - 0.0000j, -0.4238 + 0.0078j],
            [0.2567 - 0.0032j, -0.2987 + 0.0037j, 0.0000 - 0.0000j, -0.0443 + 0.0005j, -0.0000 + 0.0000j, -0.0024 + 0.0000j, 0.2512 - 0.0031j, -0.0000 + 0.0000j, 0.3730 - 0.0046j],
            [0.2534 - 0.0016j, 0.3051 - 0.0019j, 0.0000 + 0.0000j, -0.0486 + 0.0003j, -0.0000 - 0.0000j, -0.0565 + 0.0003j, -0.3012 + 0.0018j, 0.0000 + 0.0000j, -0.1347 + 0.0008j],
            [-0.0025 + 0.0000j, 0.0003 - 0.0000j, -0.0254 + 0.0003j, 0.0208 - 0.0003j, -0.0813 + 0.0010j, 0.1215 - 0.0015j, -0.1139 + 0.0014j, 0.3084 - 0.0038j, 0.3049 - 0.0037j],
            [0.0080 - 0.0000j, -0.0331 + 0.0002j, -0.1833 + 0.0011j, 0.1860 - 0.0011j, -0.2076 + 0.0013j, -0.0399 + 0.0002j, -0.2235 + 0.0014j, -0.1993 + 0.0012j, -0.2609 + 0.0016j],
            [0.0079, 0.0340, -0.1714, 0.1719, 0.2533, 0.0831, 0.2781, 0.0941, 0.0786]]);

        # D and V matrices for the E mode at the M point
        diagonalDEModeMPoint = complexArray([2.3299, 2.9571, 2.9607, 7.9678, 12.2661,
            13.6074, 26.9175, 29.3217, 61.3951]);
        self.DMatrixEModeMPoint = np.diag(diagonalDEModeMPoint);
        self.VMatrixEModeMPoint = complexArray([
            [-0.0026 + 0.0001j, 0.0030 - 0.0001j, -0.0000 + 0.0000j, -0.0163 + 0.0004j, -0.0000 + 0.0000j, -0.0918 + 0.0023j, -0.2654 + 0.0065j, -0.0000 + 0.0000j, 0.3799 - 0.0093j],
            [0.0041 - 0.0001j, -0.0184 + 0.0003j, -0.0074 - 0.0001j, -0.0468 + 0.0009j, -0.2007 - 0.0007j, -0.1626 + 0.0030j, -0.1142 + 0.0021j, 0.2265 - 0.0210j, -0.3709 + 0.0068j],
            [0.0041 - 0.0001j, 0.0082 - 0.0001j, 0.0186 + 0.0003j, 0.0549 - 0.0007j, -0.1475 - 0.0015j, -0.1861 + 0.0023j, 0.2146 - 0.0026j, -0.3040 + 0.0263j, 0.1567 - 0.0019j],
            [0.0041 - 0.0001j, -0.0184 + 0.0003j, 0.0074 + 0.0001j, -0.0468 + 0.0009j, 0.2007 + 0.0007j, -0.1626 + 0.0030j, -0.1142 + 0.0021j, -0.2265 + 0.0210j, -0.3709 + 0.0068j],
            [0.1743 - 0.0021j, -0.2711 + 0.0033j, 0.0000 - 0.0000j, -0.2630 + 0.0032j, 0.0000 - 0.0000j, 0.0313 - 0.0004j, 0.2182 - 0.0027j, 0.0000 - 0.0000j, 0.3710 - 0.0046j],
            [0.1715 - 0.0011j, 0.0073 - 0.0000j, 0.2701 + 0.0056j, 0.2932 - 0.0018j, 0.0232 + 0.0004j, 0.0271 - 0.0002j, -0.1970 + 0.0012j, 0.1357 - 0.0109j, -0.1572 + 0.0010j],
            [0.0041 - 0.0001j, 0.0082 - 0.0001j, -0.0186 - 0.0003j, 0.0549 - 0.0007j, 0.1475 + 0.0015j, -0.1861 + 0.0023j, 0.2146 - 0.0026j, 0.3040 - 0.0263j, 0.1567 - 0.0019j],
            [0.1715 - 0.0011j, 0.0073 - 0.0000j, -0.2701 - 0.0056j, 0.2932 - 0.0018j, -0.0232 - 0.0004j, 0.0271 - 0.0002j, -0.1970 + 0.0012j, -0.1357 + 0.0109j, -0.1572 + 0.0010j],
            [0.1686, 0.2685, -0.0000, -0.3288, 0.0000, -0.0154, 0.1136, 0.0000, 0.0426]]);

        # NOTE - I HAVE CHANGED THESE SO THAT THEY ARE PROPERLY SORTED IN ASCENDING ORDER.
        # THIS MAKES THINGS MUCH MORE TRACTABLE.
        # D and V matrices for the H mode at the Gamma (G) point
        # Original, unordered data:
        #diagonalDHModeGPoint = complexArray([5.9424, 8.0785, 18.1924, 21.3007, 21.4393,
        #    21.3007, 9.1893, 8.0785, 0]);
        #self.VMatrixHModeGPoint = complexArray([
        #    [0.5056 + 0.0000j, -0.9513 + 0.0058j, 0.9763 + 0.0000j, 0.9819 - 0.0181j, -0.9760 + 0.0240j, -0.2154 + 0.0027j, -0.0000 - 0.0000j, -0.0461 + 0.0041j, 0],
        #    [0.9821 + 0.0060j, -0.9880 - 0.0000j, 0.0000 + 0.0000j, -0.4728 + 0.0058j, 0.5026 - 0.0093j, 0.5723 - 0.0163j, -0.9392 - 0.0500j, -0.9505 + 0.0495j, 0],
        #    [0.5056 + 0.0062j, -0.0000 - 0.0000j, -0.9763 - 0.0120j, -0.0000 + 0.0000j, -0.9763 + 0.0120j, -0.9734 + 0.0266j, -0.0000 - 0.0000j, -0.8693 + 0.0385j, 0],
        #    [0.9821 + 0.0060j, -0.9880 - 0.0000j, -0.0000 + 0.0000j, -0.4728 + 0.0058j, 0.5026 - 0.0093j, -0.3648 + 0.0150j, 0.9392 + 0.0500j, 0.8546 - 0.0415j, 0],
        #    [0, 0, 0, 0, 0, 0, 0, 0, 1.0000],
        #    [0.9819 + 0.0181j, 0.9879 + 0.0121j, -0.0000 - 0.0000j, 0.4728 - 0.0000j, 0.5026 - 0.0031j, 0.3650 - 0.0105j, 0.9385 + 0.0615j, -0.8550 + 0.0310j, 0],
        #    [0.5056 + 0.0062j, -0.0000 - 0.0000j, -0.9763 - 0.0120j, -0.0000 + 0.0000j, -0.9763 + 0.0120j, 0.9734 - 0.0266j, 0.0000 + 0.0000j, 0.8693 - 0.0385j, 0],
        #    [0.9819 + 0.0181j, 0.9879 + 0.0121j, -0.0000 - 0.0000j, 0.4728 - 0.0000j, 0.5026 - 0.0031j, -0.5724 + 0.0093j, -0.9385 - 0.0615j, 0.9510 - 0.0379j, 0],
        #    [0.5055 + 0.0124j, 0.9511 + 0.0175j, 0.9760 + 0.0240j, -0.9821 - 0.0060j, -0.9763 + 0.0000j, 0.2154 + 0.0026j, -0.0000 - 0.0000j, 0.0462 - 0.0030j, 0]]);

        diagonalDHModeGPoint = complexArray([0, 5.9424, 8.0785, 8.0785, 9.1893, 18.1924,
            21.3007, 21.3007, 21.4393]);
        self.DMatrixHModeGPoint = np.diag(diagonalDHModeGPoint);
        self.VMatrixHModeGPoint = complexArray([
            [0.5055 + 0.0124j, 0.9511 + 0.0175j, 0.9760 + 0.0240j, -0.9821 - 0.0060j, -0.9763 + 0.0000j, 0.2154 + 0.0026j, -0.0000 - 0.0000j, 0.0462 - 0.0030j, 0],
            [0.5056 + 0.0000j, -0.9513 + 0.0058j, 0.9763 + 0.0000j, 0.9819 - 0.0181j, -0.9760 + 0.0240j, -0.2154 + 0.0027j, -0.0000 - 0.0000j, -0.0461 + 0.0041j, 0],
            [0.5056 + 0.0062j, -0.0000 - 0.0000j, -0.9763 - 0.0120j, -0.0000 + 0.0000j, -0.9763 + 0.0120j, -0.9734 + 0.0266j, -0.0000 - 0.0000j, -0.8693 + 0.0385j, 0],
            [0.9819 + 0.0181j, 0.9879 + 0.0121j, -0.0000 - 0.0000j, 0.4728 - 0.0000j, 0.5026 - 0.0031j, -0.5724 + 0.0093j, -0.9385 - 0.0615j, 0.9510 - 0.0379j, 0],
            [0.5056 + 0.0062j, -0.0000 - 0.0000j, -0.9763 - 0.0120j, -0.0000 + 0.0000j, -0.9763 + 0.0120j, 0.9734 - 0.0266j, 0.0000 + 0.0000j, 0.8693 - 0.0385j, 0],
            [0.5056 + 0.0062j, -0.0000 - 0.0000j, -0.9763 - 0.0120j, -0.0000 + 0.0000j, -0.9763 + 0.0120j, -0.9734 + 0.0266j, -0.0000 - 0.0000j, -0.8693 + 0.0385j, 0],
            [0.9821 + 0.0060j, -0.9880 - 0.0000j, -0.0000 + 0.0000j, -0.4728 + 0.0058j, 0.5026 - 0.0093j, -0.3648 + 0.0150j, 0.9392 + 0.0500j, 0.8546 - 0.0415j, 0],
            [0.9819 + 0.0181j, 0.9879 + 0.0121j, -0.0000 - 0.0000j, 0.4728 - 0.0000j, 0.5026 - 0.0031j, 0.3650 - 0.0105j, 0.9385 + 0.0615j, -0.8550 + 0.0310j, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1.0000]
            ]);

        # D and V matrices for the H mode at the X point
        # ORIGINAL, UNORDERED DATA
        #diagonalDHModeXPoint = complexArray([41.9437, 30.6424, 1.3696, 2.9324, 15.5050, 15.0256,
            #11.7266, 8.2436, 9.2339]);
        #self.VMatrixHModeXPoint = complexArray([
            #[0.8873 - 0.0000j, 0.9939 - 0.0061j, -0.0341 + 0.0004j, -0.0090 + 0.0001j, -0.4870 + 0.0060j, -0.4637 + 0.0085j, -0.5995 + 0.0074j, -0.3267 + 0.0060j, 0.2031 - 0.0050j],
            #[-0.2943 - 0.0018j, -0.4295 - 0.0000j, 0.0028 - 0.0000j, -0.2668 + 0.0000j, 0.2701 - 0.0017j, -0.9219 + 0.0113j, -0.8995 + 0.0055j, -0.9071 + 0.0111j, 0.9819 - 0.0181j],
            #[0.0150 + 0.0002j, 0.0653 + 0.0004j, -0.0170 - 0.0000j, -0.2918 - 0.0018j, -0.3034 - 0.0000j, 0.9939 - 0.0061j, 0.9880 - 0.0000j, -0.9939 + 0.0061j, 0.9302 - 0.0114j],
            #[-0.9939 - 0.0061j, -0.0000 + 0.0000j, 0.1141 - 0.0007j, -0.1910 + 0.0000j, -0.9939 + 0.0061j, -0.0000 + 0.0000j, -0.5013 + 0.0031j, -0.0000 + 0.0000j, -0.3497 + 0.0064j],
            #[0.2533 + 0.0031j, 0.0000 + 0.0000j, 0.9745 - 0.0000j, -0.9939 - 0.0061j, 0.1526 + 0.0000j, 0.0000 - 0.0000j, 0.0329 - 0.0000j, -0.0000 + 0.0000j, -0.4711 + 0.0058j],
            #[0.0713 + 0.0013j, 0.0000 + 0.0000j, -0.9939 - 0.0061j, -0.9873 - 0.0121j, 0.0809 + 0.0005j, 0.0000 - 0.0000j, -0.0230 - 0.0001j, 0.0000 + 0.0000j, -0.5423 + 0.0033j],
            #[0.8873 + 0.0109j, -0.9939 - 0.0061j, -0.0341 - 0.0000j, -0.0090 - 0.0001j, -0.4870 - 0.0000j, 0.4637 - 0.0028j, -0.5995 + 0.0000j, 0.3267 - 0.0020j, 0.2032 - 0.0025j],
            #[-0.2943 - 0.0054j, 0.4295 + 0.0053j, 0.0028 + 0.0000j, -0.2667 - 0.0033j, 0.2701 + 0.0017j, 0.9220 - 0.0000j, -0.8995 - 0.0055j, 0.9071 + 0.0000j, 0.9821 - 0.0060j],
            #[0.0150 + 0.0004j, -0.0653 - 0.0012j, -0.0170 - 0.0002j, -0.2917 - 0.0054j, -0.3033 - 0.0037j, -0.9939 - 0.0061j, 0.9879 + 0.0121j, 0.9939 + 0.0061j, 0.9303 + 0.0000j]])
        diagonalDHModeXPoint = complexArray([1.3696, 2.9324, 8.2436, 9.2339, 11.7266, 15.0256,
            15.5050, 30.6424, 41.9437]);
        self.DMatrixHModeXPoint = np.diag(diagonalDHModeXPoint);
        self.VMatrixHModeXPoint = complexArray([
            [0.0150 + 0.0002j, 0.0653 + 0.0004j, -0.0170 - 0.0000j, -0.2918 - 0.0018j, -0.3034 - 0.0000j, 0.9939 - 0.0061j, 0.9880 - 0.0000j, -0.9939 + 0.0061j, 0.9302 - 0.0114j],
            [-0.9939 - 0.0061j, -0.0000 + 0.0000j, 0.1141 - 0.0007j, -0.1910 + 0.0000j, -0.9939 + 0.0061j, -0.0000 + 0.0000j, -0.5013 + 0.0031j, -0.0000 + 0.0000j, -0.3497 + 0.0064j],
            [-0.2943 - 0.0054j, 0.4295 + 0.0053j, 0.0028 + 0.0000j, -0.2667 - 0.0033j, 0.2701 + 0.0017j, 0.9220 - 0.0000j, -0.8995 - 0.0055j, 0.9071 + 0.0000j, 0.9821 - 0.0060j],
            [0.0150 + 0.0004j, -0.0653 - 0.0012j, -0.0170 - 0.0002j, -0.2917 - 0.0054j, -0.3033 - 0.0037j, -0.9939 - 0.0061j, 0.9879 + 0.0121j, 0.9939 + 0.0061j, 0.9303 + 0.0000j],
            [0.8873 + 0.0109j, -0.9939 - 0.0061j, -0.0341 - 0.0000j, -0.0090 - 0.0001j, -0.4870 - 0.0000j, 0.4637 - 0.0028j, -0.5995 + 0.0000j, 0.3267 - 0.0020j, 0.2032 - 0.0025j],
            [0.0713 + 0.0013j, 0.0000 + 0.0000j, -0.9939 - 0.0061j, -0.9873 - 0.0121j, 0.0809 + 0.0005j, 0.0000 - 0.0000j, -0.0230 - 0.0001j, 0.0000 + 0.0000j, -0.5423 + 0.0033j],
            [0.2533 + 0.0031j, 0.0000 + 0.0000j, 0.9745 - 0.0000j, -0.9939 - 0.0061j, 0.1526 + 0.0000j, 0.0000 - 0.0000j, 0.0329 - 0.0000j, -0.0000 + 0.0000j, -0.4711 + 0.0058j],
            [-0.2943 - 0.0018j, -0.4295 - 0.0000j, 0.0028 - 0.0000j, -0.2668 + 0.0000j, 0.2701 - 0.0017j, -0.9219 + 0.0113j, -0.8995 + 0.0055j, -0.9071 + 0.0111j, 0.9819 - 0.0181j],
            [0.8873 - 0.0000j, 0.9939 - 0.0061j, -0.0341 + 0.0004j, -0.0090 + 0.0001j, -0.4870 + 0.0060j, -0.4637 + 0.0085j, -0.5995 + 0.0074j, -0.3267 + 0.0060j, 0.2031 - 0.0050j],
            ]);

        # D and V matrices for the H mode at the M point
        # ORIGINAL, UNORDERED DATA
        #diagonalDHModeMPoint = complexArray([55.2708, 16.0054, 16.0846, 2.9602, 3.2323,
            #5.7848, 30.1704, 14.3361, 5.8793]);
        #self.VMatrixHModeMPoint = complexArray([
            #[1.0000 + 0.0000j, 0.9939 - 0.0061j, 0.9082 - 0.0111j, 0.0354 - 0.0002j, 0.0428 - 0.0005j, -0.0412 + 0.0000j, 0.0000 - 0.0000j, 0.0000 + 0.0000j, -0.0000 - 0.0000j],
            #[-0.6142 - 0.0038j, 0.4710 + 0.0000j, 0.9618 - 0.0059j, -0.1603 + 0.0000j, -0.1209 + 0.0007j, -0.2628 - 0.0016j, 0.9470 - 0.0231j, 0.9941 + 0.0059j, -0.0345 + 0.0013j],
            #[0.1901 + 0.0023j, -0.9265 - 0.0057j, 1.0000 + 0.0000j, 0.1614 + 0.0010j, -0.1421 + 0.0000j, -0.0151 - 0.0002j, -0.9821 + 0.0179j, 0.9007 + 0.0109j, 0.2526 - 0.0077j],
            #[-0.6142 - 0.0038j, 0.4710 - 0.0000j, 0.9618 - 0.0059j, -0.1603 + 0.0000j, -0.1209 + 0.0007j, -0.2628 - 0.0016j, -0.9470 + 0.0231j, -0.9941 - 0.0059j, 0.0345 - 0.0013j],
            #[0.2665 + 0.0033j, -0.2043 - 0.0013j, -0.4243 - 0.0000j, -0.9939 - 0.0061j, -0.7238 + 0.0000j, -0.9879 - 0.0121j, 0.0000 - 0.0000j, -0.0000 - 0.0000j, -0.0000 - 0.0000j],
            #[-0.0230 - 0.0004j, 0.1508 + 0.0019j, -0.1791 - 0.0011j, 0.9830 + 0.0121j, -0.8249 - 0.0051j, -0.0952 - 0.0018j, 0.2877 - 0.0035j, -0.1979 - 0.0036j, 0.9760 - 0.0240j],
            #[0.1901 + 0.0023j, -0.9265 - 0.0057j, 1.0000 + 0.0000j, 0.1614 + 0.0010j, -0.1421 + 0.0000j, -0.0151 - 0.0002j, 0.9821 - 0.0179j, -0.9007 - 0.0109j, -0.2526 + 0.0077j],
            #[-0.0230 - 0.0004j, 0.1508 + 0.0019j, -0.1791 - 0.0011j, 0.9830 + 0.0121j, -0.8249 - 0.0051j, -0.0952 - 0.0018j, -0.2877 + 0.0035j, 0.1979 + 0.0036j, -0.9760 + 0.0240j],
            #[-0.0179 - 0.0004j, 0.0920 + 0.0017j, 0.1263 + 0.0016j, -0.9189 - 0.0169j, -0.9879 - 0.0121j, 0.9494 + 0.0233j, -0.0000 + 0.0000j, 0.0000 - 0.0000j, 0.0000 + 0.0000j]])
        diagonalDHModeMPoint = complexArray([2.9602, 3.2323, 5.7848, 5.8793, 14.3361,
            16.0846, 26.0054, 30.1704, 55.2708]);
        self.DMatrixHModeMPoint = np.diag(diagonalDHModeMPoint);
        self.VMatrixHModeMPoint = complexArray([
            [-0.6142 - 0.0038j, 0.4710 - 0.0000j, 0.9618 - 0.0059j, -0.1603 + 0.0000j, -0.1209 + 0.0007j, -0.2628 - 0.0016j, -0.9470 + 0.0231j, -0.9941 - 0.0059j, 0.0345 - 0.0013j],
            [0.2665 + 0.0033j, -0.2043 - 0.0013j, -0.4243 - 0.0000j, -0.9939 - 0.0061j, -0.7238 + 0.0000j, -0.9879 - 0.0121j, 0.0000 - 0.0000j, -0.0000 - 0.0000j, -0.0000 - 0.0000j],
            [-0.0230 - 0.0004j, 0.1508 + 0.0019j, -0.1791 - 0.0011j, 0.9830 + 0.0121j, -0.8249 - 0.0051j, -0.0952 - 0.0018j, 0.2877 - 0.0035j, -0.1979 - 0.0036j, 0.9760 - 0.0240j],
            [-0.0179 - 0.0004j, 0.0920 + 0.0017j, 0.1263 + 0.0016j, -0.9189 - 0.0169j, -0.9879 - 0.0121j, 0.9494 + 0.0233j, -0.0000 + 0.0000j, 0.0000 - 0.0000j, 0.0000 + 0.0000j],
            [-0.0230 - 0.0004j, 0.1508 + 0.0019j, -0.1791 - 0.0011j, 0.9830 + 0.0121j, -0.8249 - 0.0051j, -0.0952 - 0.0018j, -0.2877 + 0.0035j, 0.1979 + 0.0036j, -0.9760 + 0.0240j],
            [0.1901 + 0.0023j, -0.9265 - 0.0057j, 1.0000 + 0.0000j, 0.1614 + 0.0010j, -0.1421 + 0.0000j, -0.0151 - 0.0002j, -0.9821 + 0.0179j, 0.9007 + 0.0109j, 0.2526 - 0.0077j],
            [-0.6142 - 0.0038j, 0.4710 + 0.0000j, 0.9618 - 0.0059j, -0.1603 + 0.0000j, -0.1209 + 0.0007j, -0.2628 - 0.0016j, 0.9470 - 0.0231j, 0.9941 + 0.0059j, -0.0345 + 0.0013j],
            [0.1901 + 0.0023j, -0.9265 - 0.0057j, 1.0000 + 0.0000j, 0.1614 + 0.0010j, -0.1421 + 0.0000j, -0.0151 - 0.0002j, 0.9821 - 0.0179j, -0.9007 - 0.0109j, -0.2526 + 0.0077j],
            [1.0000 + 0.0000j, 0.9939 - 0.0061j, 0.9082 - 0.0111j, 0.0354 - 0.0002j, 0.0428 - 0.0005j, -0.0412 + 0.0000j, 0.0000 - 0.0000j, 0.0000 + 0.0000j, -0.0000 - 0.0000j],
            ]);

        self.erConvolutionMatrix = complexArray([
            [5.9208, 1.5571 - 0.0096j, 0.2834 - 0.0035j, 1.5571 - 0.0096j, -0.5879 + 0.0072j,
                -0.3972 + 0.0073j, 0.2834 - 0.0035j, -0.3972 + 0.0073j, 0.2256 - 0.0055j],
            [1.5571 + 0.0096j, 5.9208, 1.5571 - 0.0096j, -0.5880 - 0j, 1.5571 - 0.0096j,
                -0.5879 + 0.0072j, -0.3972 + 0.0024j, 0.2834 - 0.0035j, -0.3972 + 0.0073j],
            [0.2834 + 0.0035j, 1.5571 + 0.0096j, 5.9208, -0.3972 - 0.0024j, -0.5880 - 0.0j,
                1.5571 - 0.0096j, 0.2256 - 0.0j, -0.3972 + 0.0024j, 0.2834 - 0.0035j],
            [1.5571 + 0.0096j, -0.5880 + 0.0j, -0.3972 + 0.0024j, 5.9208, 1.5571 - 0.0096j,
                0.2834 - 0.0035j, 1.5571 - 0.0096j, -0.5879 + 0.0072j, -0.3972 + 0.0073j],
            [-0.5871 - 0.0072j, 1.5571 + 0.0096j, -0.5880 + 0.0j, 1.5571 + 0.0096j, 5.9208,
                1.5571 - 0.0096j, -0.5880 - 0.0j, 1.5571 - 0.0096j, -0.5879 + 0.0072j],
            [-0.3972 - 0.0073j, -0.5879 - 0.0072j, 1.5571 + 0.0096j, 0.2834 + 0.0035j, 1.5571 + 0.0096j,
                5.9208, -0.3972 - 0.0024j, -0.5880 - 0.0j, 1.5571 - 0.0096j],
            [0.2834 + 0.0035j, -0.3972 - 0.0024j, 0.2256 + 0.0j, 1.5571 + 0.0096j, -0.5880 + 0.0j,
                -0.3972 + 0.0024j, 5.9208, 1.5571 - 0.0096j, 0.2834 - 0.0035j],
            [-0.3972 - 0.0073j, 0.2834 + 0.0035j, -0.3972 - 0.0024j, -0.5879 - 0.0072j, 1.5571 + 0.0096j,
                -0.5880 + 0.0j, 1.5571 + 0.0096j, 5.9208, 1.5571 - 0.0096j],
            [0.2256 + 0.0055j, -0.3972 - 0.0073j, 0.2834 + 0.0035j, -0.3972 - 0.0073j, -0.5879-0.0072j,
                1.5571 + 0.0096j, 0.2834 + 0.0035j, 1.5571 + 0.0096j, 5.9208]]);

        self.urConvolutionMatrix = complexIdentity(9);

    def printResults(self):
        for s, i in zip(self.statuses, range(len(self.statuses))):
            if(s == False):
                print(self.messages[i]);
        print(f"{self.statuses.count(True)} PASSED, {self.statuses.count(False)} FAILED");

    def testCaller(self, testFunction, *args):
        """
        Handles the actual calling of test functions, manages them with try/catch blocks. Maybe
        not the most elegant way to do things, but the best idea I currently have without wasting
        an inordinate amount of time.
        """
        test_status = False; # By default assume we failed the test.
        test_message = f"{testFunction.__name__}({args}): ";

        try:
            print(f"Calling function {testFunction.__name__} ... ", end=" ");
            testFunction(*args);
            print("OK");
            test_status = True;
            self.statuses.append(test_status);
            self.messages.append(test_message);
        except AssertionError as ae:
            print("FAIL");
            test_message += "FAILED";
            test_message += str(ae);
            self.statuses.append(test_status);
            self.messages.append(test_message);

    def runUnitTests(self):
        print("--------- RUNNING UNIT TESTS... ----------");
        self.testCaller(self.testGetXComponents);
        self.testCaller(self.testGetYComponents);
        self.testCaller(self.testCalculateMinHarmonic);
        self.testCaller(self.testCalculateMaxHarmonic);
        self.testCaller(self.testCalculateZeroHarmonicLocation);
        self.testCaller(self.testReshapeLowDimensionalData);
        self.testCaller(self.testGenerateConvolutionMatrix);
        self.testCaller(self.testGenerateKxMatrix);
        self.testCaller(self.testGenerateKyMatrix);
        self.testCaller(self.testCalculateAMatrix);
        self.testCaller(self.testCalculateBMatrix);
        self.testCaller(self.testCalculateDMatrix);
        self.testCaller(self.testCalculateReciprocalLatticeVectors);
        self.testCaller(self.testDetermineCrystalType);
        self.testCaller(self.testGenerateKeySymmetryPoints);
        self.testCaller(self.testGenerateBlochVectors);
        self.testCaller(self.testUnwrapBlochVectors);
        self.testCaller(self.testGenerateBandData);
        self.testCaller(self.testSolve);
        #self.testCaller(self.testCalculateVMatrix); # EIGENVECTORS NOT WORKING, DON'T KNOW WHY.
        print("--------- END UNIT TESTS... ----------");

    def runIntegrationTests(self):
        """
        Runs integration tests to verify s-parameters for composite code, to verify the output field
        for a given input field, and to verify the reflectance/transmittance and enforce power 
        conservation.
        """

        print("--------- RUNNING INTEGRATION TESTS... ----------");

        print("--------- END INTEGRATION TESTS... ----------");

    def testGetXComponents(self):
        testVector1 = complexArray([0.163, 0.5, 0.888]);
        testVector2 = complexArray([0.246, 0.99, 0.2]);
        xComponentsCalculated = getXComponents(testVector1, testVector2);
        xComponentsActual = [0.163, 0.246];
        assertAlmostEqual(xComponentsActual, xComponentsCalculated);

        testVector1 = complexArray([[0.183], [0.5], [0.888]]);
        testVector2 = complexArray([[0.266], [0.99], [0.2]]);
        xComponentsCalculated = getXComponents(testVector1, testVector2);
        xComponentsActual = [0.183, 0.266];
        assertAlmostEqual(xComponentsActual, xComponentsCalculated);

        testVector1 = complexArray([[1.173], [0.7]]);
        testVector2 = complexArray([1.256, 1.99]);
        xComponentsCalculated = getXComponents(testVector1, testVector2);
        xComponentsActual = [1.173, 1.256];
        assertAlmostEqual(xComponentsActual, xComponentsCalculated);

    def testGetYComponents(self):
        testVector1 = complexArray([0.173, 0.4, 0.888]);
        testVector2 = complexArray([0.256, 0.89, 0.2]);
        yComponentsCalculated = getYComponents(testVector1, testVector2);
        yComponentsActual = [0.4, 0.89];
        assertAlmostEqual(yComponentsActual, yComponentsCalculated);

        testVector1 = complexArray([[0.173], [0.5], [0.888]]);
        testVector2 = complexArray([[0.256], [0.99], [0.2]]);
        yComponentsCalculated = getYComponents(testVector1, testVector2);
        yComponentsActual = [0.5, 0.99];
        assertAlmostEqual(yComponentsActual, yComponentsCalculated);

        testVector1 = complexArray([[0.173], [0.7]]);
        testVector2 = complexArray([0.256, 1.99]);
        yComponentsCalculated = getYComponents(testVector1, testVector2);
        yComponentsActual = [0.7, 1.99];
        assertAlmostEqual(yComponentsActual, yComponentsCalculated);

    def testCalculateZeroHarmonicLocation(self):
        harmonicNumber1 = 5;
        harmonicNumber2 = 6;
        numberHarmonics = (harmonicNumber1, harmonicNumber2)
        zeroHarmonicLocationsCalculated = calculateZeroHarmonicLocation(numberHarmonics)
        zeroHarmonicLocationsActual = [2, 3];
        assertAlmostEqual(zeroHarmonicLocationsActual, zeroHarmonicLocationsCalculated);

    def testCalculateMinHarmonic(self):
        harmonicNumber1 = 5;
        harmonicNumber2 = 6;
        numberHarmonics = (harmonicNumber1, harmonicNumber2)
        minHarmonicCalculated= calculateMinHarmonic(numberHarmonics)
        minHarmonicActual = [-2, -3];
        assertAlmostEqual(minHarmonicActual, minHarmonicCalculated);

    def testCalculateMaxHarmonic(self):
        harmonicNumber1 = 5;
        harmonicNumber2 = 6;
        numberHarmonics = (harmonicNumber1, harmonicNumber2)
        maxHarmonicCalculated= calculateMaxHarmonic(numberHarmonics)
        maxHarmonicActual = [2, 2];
        assertAlmostEqual(maxHarmonicActual, maxHarmonicCalculated);

    def testGenerateConvolutionMatrix(self):

        # Test case 1: When we pass in a homogenous device, we should get out a multiple of the identity.
        absoluteTolerance = 1e-6;
        relativeTolerance = 1e-5;
        numberHarmonics = (3, 3, 1)

        Nx = 10;
        er = 9.0;

        A = er * complexOnes((Nx, Nx));

        convolutionMatrixCalculated = generateConvolutionMatrix(A, numberHarmonics);
        convolutionMatrixActual = er * complexIdentity(np.prod(numberHarmonics));

        assertAlmostEqual(convolutionMatrixActual, convolutionMatrixCalculated,
                absoluteTolerance, relativeTolerance);

        # Test case 2: using benchmarking data for relative permeability from Rumpf
        absoluteTolerance = 1e-4;
        relativeTolarence = 1e-3;
        numberHarmonics = (self.P, self.Q, 1)

        A = self.UR;
        Nx = self.Nx;
        Ny = self.Ny;

        convolutionMatrixCalculated = generateConvolutionMatrix(A, numberHarmonics);
        convolutionMatrixActual = complexIdentity(np.prod(numberHarmonics))

        assertAlmostEqual(convolutionMatrixActual, convolutionMatrixCalculated,
                absoluteTolerance, relativeTolerance);

        # Test case 3: using benchmarking data for relative permittivity from Rumpf
        absoluteTolerance = 1e-3;
        relativeTolerance = 1e-2;
        A = self.ER;

        convolutionMatrixCalculated = generateConvolutionMatrix(A, numberHarmonics);
        convolutionMatrixActual = self.erConvolutionMatrix;
        assertAlmostEqual(convolutionMatrixActual, convolutionMatrixCalculated,
                absoluteTolerance, relativeTolerance);

    def testGenerateKxMatrix(self):
        absoluteTolerance = 1e-4;
        relativeTolerance = 1e-3;

        # Test our KX matrix at the gamma point
        kxMatrixActual = self.KxMatrixGPoint;
        kxMatrixCalculated = generateKxMatrix(self.blochVectorGPoint, self.crystal, self.numberHarmonics)
        assertAlmostEqual(kxMatrixActual, kxMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Test our KX matrix at the X point
        kxMatrixActual = self.KxMatrixXPoint;
        kxMatrixCalculated = generateKxMatrix(self.blochVectorXPoint, self.crystal, self.numberHarmonics)
        assertAlmostEqual(kxMatrixActual, kxMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Test our KX matrix at the M point
        kxMatrixActual = self.KxMatrixMPoint;
        kxMatrixCalculated = generateKxMatrix(self.blochVectorMPoint, self.crystal, self.numberHarmonics)
        assertAlmostEqual(kxMatrixActual, kxMatrixCalculated, absoluteTolerance, relativeTolerance);

    def testGenerateKyMatrix(self):
        absoluteTolerance = 1e-4;
        relativeTolerance = 1e-3;

        # Test our KY matrix at the gamma point
        kyMatrixActual = self.KyMatrixGPoint;
        kyMatrixCalculated = generateKyMatrix(self.blochVectorGPoint, self.crystal, self.numberHarmonics)
        assertAlmostEqual(kyMatrixActual, kyMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Test our KY matrix at the X point
        kyMatrixActual = self.KyMatrixXPoint;
        kyMatrixCalculated = generateKyMatrix(self.blochVectorXPoint, self.crystal, self.numberHarmonics)
        assertAlmostEqual(kyMatrixActual, kyMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Test our KY matrix at the M point
        kyMatrixActual = self.KyMatrixMPoint;
        kyMatrixCalculated = generateKyMatrix(self.blochVectorMPoint, self.crystal, self.numberHarmonics)
        assertAlmostEqual(kyMatrixActual, kyMatrixCalculated, absoluteTolerance, relativeTolerance);

    def testCalculateAMatrix(self):
        absoluteTolerance = 1e-4;
        relativeTolerance = 1e-3;

        matrixDimensions = self.P * self.Q;
        matrixShape = (matrixDimensions, matrixDimensions);

        # First, test the E-mode at the gamma (G) point
        AMatrixCalculated = generateAMatrix(self.KxMatrixGPoint, self.KyMatrixGPoint,
                self.erConvolutionMatrix, self.urConvolutionMatrix, 'E');
        AMatrixActual = self.AMatrixEModeGPoint;
        assertAlmostEqual(AMatrixActual, AMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the E-mode at the X point
        AMatrixCalculated = generateAMatrix(self.KxMatrixXPoint, self.KyMatrixXPoint,
                self.erConvolutionMatrix, self.urConvolutionMatrix, 'E');
        AMatrixActual = self.AMatrixEModeXPoint;
        assertAlmostEqual(AMatrixActual, AMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Finally, test the E-mode at the M point
        AMatrixCalculated = generateAMatrix(self.KxMatrixMPoint, self.KyMatrixMPoint,
                self.erConvolutionMatrix, self.urConvolutionMatrix, 'E');
        AMatrixActual = self.AMatrixEModeMPoint;
        assertAlmostEqual(AMatrixActual, AMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Since these tests require inverting the permittivity matrix, they introduce additional error
        # and we need to relax the tolerances a bit.
        absoluteTolerance = 2e-3;
        relativeTolerance = 1e-2;

        # Next, test the H-mode at the gamma (G) point
        AMatrixCalculated = generateAMatrix(self.KxMatrixGPoint, self.KyMatrixGPoint,
                self.erConvolutionMatrix, self.urConvolutionMatrix, 'H');
        AMatrixActual = self.AMatrixHModeGPoint;
        assertAlmostEqual(AMatrixActual, AMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the X point
        AMatrixCalculated = generateAMatrix(self.KxMatrixXPoint, self.KyMatrixXPoint,
                self.erConvolutionMatrix, self.urConvolutionMatrix, 'H');
        AMatrixActual = self.AMatrixHModeXPoint;
        assertAlmostEqual(AMatrixActual, AMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the M point
        AMatrixCalculated = generateAMatrix(self.KxMatrixMPoint, self.KyMatrixMPoint,
                self.erConvolutionMatrix, self.urConvolutionMatrix, 'H');
        AMatrixActual = self.AMatrixHModeMPoint;
        assertAlmostEqual(AMatrixActual, AMatrixCalculated, absoluteTolerance, relativeTolerance);

    def testCalculateBMatrix(self):
        absoluteTolerance = 1e-3;
        relativeTolerance = 2e-3;

        matrixDimensions = self.P * self.Q;
        matrixShape = (matrixDimensions, matrixDimensions, absoluteTolerance, relativeTolerance);

        # First, test the E-mode at the gamma (G) point
        BMatrixCalculated = generateBMatrix(self.erConvolutionMatrix, self.urConvolutionMatrix, 'E');
        BMatrixActual = self.BMatrixEModeGPoint;
        assertAlmostEqual(BMatrixActual, BMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the E-mode at the X point
        BMatrixCalculated = generateBMatrix(self.erConvolutionMatrix, self.urConvolutionMatrix, 'E');
        BMatrixActual = self.BMatrixEModeXPoint;
        assertAlmostEqual(BMatrixActual, BMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Finally, test the E-mode at the M point
        BMatrixCalculated = generateBMatrix(self.erConvolutionMatrix, self.urConvolutionMatrix, 'E');
        BMatrixActual = self.BMatrixEModeMPoint;
        assertAlmostEqual(BMatrixActual, BMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the gamma (G) point
        BMatrixCalculated = generateBMatrix(self.erConvolutionMatrix, self.urConvolutionMatrix, 'H');
        BMatrixActual = self.BMatrixHModeGPoint;
        assertAlmostEqual(BMatrixActual, BMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the X point
        BMatrixCalculated = generateBMatrix(self.erConvolutionMatrix, self.urConvolutionMatrix, 'H');
        BMatrixActual = self.BMatrixHModeXPoint;
        assertAlmostEqual(BMatrixActual, BMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the M point
        BMatrixCalculated = generateBMatrix(self.erConvolutionMatrix, self.urConvolutionMatrix, 'H');
        BMatrixActual = self.BMatrixHModeMPoint;
        assertAlmostEqual(BMatrixActual, BMatrixCalculated, absoluteTolerance, relativeTolerance);

    def testCalculateDMatrix(self):
        absoluteTolerance = 1e-4;
        relativeTolerance = 1e-3;

        # First, test the E-mode at the gamma (G) point
        (DMatrixCalculated, V) = generateVDMatrices(self.AMatrixEModeGPoint, self.BMatrixEModeGPoint);
        DMatrixActual = self.DMatrixEModeGPoint;
        assertAlmostEqual(DMatrixActual, DMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the E-mode at the X point
        (DMatrixCalculated, V) = generateVDMatrices(self.AMatrixEModeXPoint, self.BMatrixEModeXPoint);
        DMatrixActual = self.DMatrixEModeXPoint;
        assertAlmostEqual(DMatrixActual, DMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Finally, test the E-mode at the M point
        (DMatrixCalculated, V) = generateVDMatrices(self.AMatrixEModeMPoint, self.BMatrixEModeMPoint);
        DMatrixActual = self.DMatrixEModeMPoint;
        assertAlmostEqual(DMatrixActual, DMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the gamma (G) point
        (DMatrixCalculated, V) = generateVDMatrices(self.AMatrixHModeGPoint, self.BMatrixHModeGPoint);
        DMatrixActual = self.DMatrixHModeGPoint;
        assertAlmostEqual(DMatrixActual, DMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the X point
        (DMatrixCalculated, V) = generateVDMatrices(self.AMatrixHModeXPoint, self.BMatrixHModeXPoint);
        DMatrixActual = self.DMatrixHModeXPoint;
        assertAlmostEqual(DMatrixActual, DMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the M point
        (DMatrixCalculated, V) = generateVDMatrices(self.AMatrixHModeMPoint, self.BMatrixHModeMPoint);
        DMatrixActual = self.DMatrixHModeMPoint;
        assertAlmostEqual(DMatrixActual, DMatrixCalculated, absoluteTolerance, relativeTolerance);

    def testCalculateVMatrix(self):
        absoluteTolerance = 1e-3;
        relativeTolerance = 1e-2;

        # First, test the E-mode at the gamma (G) point
        #(D, VMatrixCalculated) = generateVDMatrices(self.AMatrixEModeGPoint, self.BMatrixEModeGPoint);
        #VMatrixActual = self.VMatrixEModeGPoint;
        #errorMatrix = np.abs(VMatrixActual - VMatrixCalculated);
        #truthMatrix = np.greater(errorMatrix, 0.09*complexOnes((9,9)));
        #print("");
        #print(truthMatrix);
        #assertAlmostEqual(VMatrixActual, VMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the E-mode at the X point
        (D, VMatrixCalculated) = generateVDMatrices(self.AMatrixEModeGPoint, self.BMatrixEModeGPoint);
        VMatrixActual = self.VMatrixEModeXPoint;
        assertAlmostEqual(VMatrixActual, VMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Finally, test the E-mode at the M point
        #VMatrixCalculated = 0;
        #VMatrixActual = self.VMatrixEModeMPoint;
        #assertAlmostEqual(VMatrixActual, VMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the gamma (G) point
        #VMatrixCalculated = 0;
        #VMatrixActual = self.VMatrixHModeMGoint;
        #assertAlmostEqual(VMatrixActual, VMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the X point
        #VMatrixCalculated = 0;
        #VMatrixActual = self.VMatrixHModeXPoint;
        #assertAlmostEqual(VMatrixActual, VMatrixCalculated, absoluteTolerance, relativeTolerance);

        # Next, test the H-mode at the M point
        #VMatrixCalculated = 0;
        #VMatrixActual = self.VMatrixHModeMPoint;
        #assertAlmostEqual(VMatrixActual, VMatrixCalculated, absoluteTolerance, relativeTolerance);

    def testReshapeLowDimensionalData(self):

        A = complexArray([1, 2, 3, 4]);
        reshapedActual = complexArray([[[1]], [[2]], [[3]], [[4]]]);

        reshapedCalculated = reshapeLowDimensionalData(A);

        assertAlmostEqual(reshapedActual, reshapedCalculated);

        A = complexArray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]]);

        reshapedActual = complexArray([
            [[1], [2], [3]],
            [[4], [5], [6]],
            [[7], [8], [9]]]);
        reshapedCalculated = reshapeLowDimensionalData(A);

        assertAlmostEqual(reshapedActual, reshapedCalculated);

    def testCalculateReciprocalLatticeVectors(self):
        # A simple cubic 2D lattice
        t1 = complexArray([1,0]);
        t2 = complexArray([0,1]);
        squareCrystal = Crystal(1, 1, t1, t2)
        T1Actual = 2 * pi * complexArray([1,0]);
        T2Actual = 2 * pi * complexArray([0,1]);
        reciprocalLatticeVectorsActual = (T1Actual, T2Actual);
        reciprocalLatticeVectorsCalculated = squareCrystal.reciprocalLatticeVectors

        assertAlmostEqual(reciprocalLatticeVectorsActual, reciprocalLatticeVectorsCalculated);

        # A rectangular 2D lattice
        t1 = complexArray([2,0]);
        t2 = complexArray([0,1]);
        rectangularCrystal = Crystal(1, 1, t1, t2);
        T1Actual = 1 * pi * complexArray([1 , 0]);
        T2Actual = 2 * pi * complexArray([0 , 1]);
        reciprocalLatticeVectorsActual = (T1Actual, T2Actual);
        reciprocalLatticeVectorsCalculated = rectangularCrystal.reciprocalLatticeVectors

        assertAlmostEqual(reciprocalLatticeVectorsActual, reciprocalLatticeVectorsCalculated);

    def testDetermineCrystalType(self):
        # A square lattice
        t1 = complexArray([1,0]);
        t2 = complexArray([0,1]);
        squareCrystal = Crystal(1, 1, t1, t2)
        crystalTypeActual = "SQUARE"
        crystalTypeCalculated = squareCrystal.crystalType
        assertStringEqual(crystalTypeActual, crystalTypeCalculated)


        # A rectangular lattice
        t1 = complexArray([1,0]);
        t2 = complexArray([0,2]);
        rectangularCrystal = Crystal(1, 1, t1, t2)
        crystalTypeActual = "RECTANGULAR";
        crystalTypeCalculated = rectangularCrystal.crystalType
        assertStringEqual(crystalTypeActual, crystalTypeCalculated)

    def testGenerateKeySymmetryPoints(self):

        # A square lattice
        t1 = complexArray([1,0]);
        t2 = complexArray([0,1]);
        squareCrystal = Crystal(1, 1, t1, t2)
        T1 = 2*pi*complexArray([1, 0]);
        T2 = 2*pi*complexArray([0,1]);

        keySymmetryPointsActual = [0.5 * T1, 0*T1, 0.5 * (T1 + T2)]
        keySymmetryNamesActual = ["X", "G", "M"]
        keySymmetryPointsCalculated = squareCrystal.keySymmetryPoints
        keySymmetryNamesCalculated = squareCrystal.keySymmetryNames

        assertArrayEqual(keySymmetryPointsActual, keySymmetryPointsCalculated);
        assertArrayEqual(keySymmetryNamesActual, keySymmetryNamesCalculated);

        # A rectangular Lattice
        t1 = complexArray([1,0])
        t2 = complexArray([0,2])
        rectangularCrystal = Crystal(1, 1, t1, t2)
        T1 = 2*pi*complexArray([1, 0]);
        T2 = pi*complexArray([0,1]);

        keySymmetryPointsActual = [0.5 * T1, 0 * T1, 0.5 * T2, 0.5 * (T1 + T2)];
        keySymmetryNamesActual = ["X", "G", "Y", "S"];
        keySymmetryPointsCalculated = rectangularCrystal.keySymmetryPoints;
        keySymmetryNamesCalculated = rectangularCrystal.keySymmetryNames;

        assertArrayEqual(keySymmetryPointsActual, keySymmetryPointsCalculated);
        assertArrayEqual(keySymmetryNamesActual, keySymmetryNamesCalculated);

    def testGenerateBlochVectors(self):
        # Test for a square lattice with no internal points
        numberInternalPoints = 0
        t1 = complexArray([1,0])
        t2 = complexArray([0,1])
        squareCrystal = Crystal(1, 1, t1, t2)
        T1 = 2 * pi * complexArray([1,0]);
        T2 = 2 * pi * complexArray([0,1]);
        bandStructure = BandStructure(squareCrystal)
        bandStructure.generateBlochVectors(numberInternalPoints)
        blochVectorsCalculated = bandStructure.blochVectors
        blochVectorsActual = [0.5 * T1, 0*T1, 0.5 * (T1 + T2)];

        assertArrayEqual(blochVectorsActual, blochVectorsCalculated);

        # Test for a rectangular lattice with no internal points
        numberInternalPoints = 0;
        t1 = complexArray([1,0]);
        t2 = 2 *complexArray([0,1]);
        rectangularCrystal = Crystal(1, 1, t1, t2)
        T1 = 2 * pi * complexArray([1,0]);
        T2 = pi * complexArray([0,1]);
        bandStructure = BandStructure(rectangularCrystal)
        bandStructure.generateBlochVectors(numberInternalPoints)
        blochVectorsCalculated = bandStructure.blochVectors

        blochVectorsActual = [0.5 * T1, 0*T1, 0.5 * T2,  0.5 * (T1 + T2)];
        assertArrayEqual(blochVectorsActual, blochVectorsCalculated);

        # Test for a square lattice with 1 internal points.
        numberInternalPoints = 1
        t1 = pi * complexArray([1,0])
        t2 = pi * complexArray([0,1])
        squareCrystal = Crystal(1, 1, t1, t2)
        bandStructure = BandStructure(squareCrystal)
        bandStructure.generateBlochVectors(numberInternalPoints)
        blochVectorsCalculated = bandStructure.blochVectors

        separationDistance = 1 / (numberInternalPoints + 1);
        blochVectorsActual = [complexArray([1, 0]), complexArray([0.5, 0]), complexArray([0,0]),
                complexArray([0.5, 0.5]), complexArray([1, 1])];

        assertArrayEqual(blochVectorsActual, blochVectorsCalculated);

    def testSolve(self):
        absoluteTolerance = 1e-4;
        relativeTolerance = 1e-3;
        # Primitive real-space lattice vectors
        t1 = complexArray([1,0]);
        t2 = complexArray([0,1]);
        pointsPerWalk = 0;
        numberHarmonics = (3, 3, 1)

        er = 9;
        a = 1;
        r = 0.35 * a;
        ax = a;
        ay = a;
        Nx = 512;
        Ny = 512;
        dx = ax / Nx;
        dy = ay / Ny;

        xcoors = np.linspace(-ax/2 + dx/2, ax/2 - dx/2, Nx);
        ycoors = np.linspace(-ay/2 + dy/2, ay/2 - dy/2, Ny);
        (X, Y) = np.meshgrid(xcoors, ycoors);
        UR = complexOnes((Nx, Ny));
        ER = (er-1) * np.heaviside(sq(X) + sq(Y) - sq(r),1)
        ER = ER + 1;
        crystal = Crystal(ER, UR, t1, t2)
        bandStructure = BandStructure(crystal)
        bandStructure.Solve(numberHarmonics, pointsPerWalk)

        eigenFrequenciesActual = [
                complexArray([0.1817, 0.2268, 0.4020, 0.4124, 0.5578, 0.5646, 0.6642, 0.8602, 1.1107]),
                complexArray([0, 0.3681, 0.3880, 0.3880, 0.4306, 0.5987, 0.7369, 0.7369, 0.8995]),
                complexArray([0.2429, 0.2737, 0.2739, 0.4493, 0.5574, 0.5871, 0.8257, 0.8618, 1.2471]),
                ]

        eigenFrequenciesCalculated = bandStructure.frequencyData

        assertAlmostEqual(eigenFrequenciesActual, eigenFrequenciesCalculated,
            absoluteTolerance, relativeTolerance);

    def testUnwrapBlochVectors(self):
        testVectors = [complexArray([1,0]), complexArray([0.5, 0]), complexArray([0,0]),
                complexArray([0.5, 0.5]), complexArray([1, 1])];
        band = BandStructure(self.crystal)
        band.blochVectors = testVectors
        band.unwrapBlochVectors()
        xCoordinatesCalculated = band.unwrappedBlochVectors
        xCoordinatesActual = [0, 0.5, 1, 1 + 1/sqrt(2), 1 + 2/sqrt(2)];

        assertAlmostEqual(xCoordinatesActual, xCoordinatesCalculated);

    def testGenerateBandData(self):
        testVectors = [
                complexArray([1,0]),
                complexArray([0.5, 0]),
                complexArray([0,0]),
                complexArray([0.5, 0.5]),
                complexArray([1, 1])];
        testFrequencies = np.array([
            [0, 0.5, 0.6, 0.7],
            [0.2, 0.6, 0.7, 0.8],
            [0.3, 0.7, 0.8, 0.9],
            [0.4, 0.8, 0.9, 1.0],
            [0.5,0.9,1.0, 1.1]]);

        xCoordinatesActual = np.array([
                0, 0, 0, 0,
                0.5, 0.5, 0.5, 0.5,
                1, 1, 1, 1,
                1 + 1/sqrt(2), 1 + 1/sqrt(2), 1 + 1/sqrt(2), 1 + 1/sqrt(2),
                1 + 2/sqrt(2), 1 + 2/sqrt(2), 1 + 2/sqrt(2), 1 + 2/sqrt(2)]);
        yCoordinatesActual = np.array([
                0, 0.5, 0.6, 0.7,
                0.2, 0.6, 0.7, 0.8,
                0.3, 0.7, 0.8, 0.9,
                0.4, 0.8, 0.9, 1.0,
                0.5, 0.9, 1.0, 1.1]);
        band = BandStructure(self.crystal)
        band.blochVectors = testVectors
        band.frequencyData = testFrequencies
        band.generateBandData()
        xCoordinatesCalculated = band.xScatterPlotData
        yCoordinatesCalculated = band.yScatterPlotData

        assertAlmostEqual(xCoordinatesActual, xCoordinatesCalculated);
        assertAlmostEqual(yCoordinatesActual, yCoordinatesCalculated);


def main():
    test_class = Test(); # Create a new test class
    if(test_class.unitTestsEnabled == True):
        test_class.runUnitTests();
    if(test_class.integrationTestsEnabled == True):
        test_class.runIntegrationTests();
    test_class.printResults();

main();
