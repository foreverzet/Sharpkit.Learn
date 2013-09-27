// -----------------------------------------------------------------------
// <copyright file="RidgeTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.LinearModel
{
    using System.Collections.Generic;
    using System.Linq;
    using System;
    using MathNet.Numerics.Distributions;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Learn.Datasets;
    using Learn.LinearModel;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// Tests for <see cref="RidgeRegression"/> and <see cref="RidgeClassifier{TLabel}"/>.
    /// </summary>
    [TestClass]
    public class RidgeTest
    {
        private Matrix<double> xDiabetes;
        private Matrix<double> yDiabetes;

        private Matrix<double> xIris;
        private int[] yIris;

        [TestInitialize]
        public void TestInitialize()
        {
            var diabetes = DiabetesDataset.Load();
            xDiabetes = diabetes.Data.SubMatrix(0, 200, 0, diabetes.Data.ColumnCount);
            yDiabetes = MatrixExtensions.ToColumnMatrix(diabetes.Target.SubVector(0, 200));
            //ind = np.arange(X_diabetes.shape[0])
            //Random rng = new Random(0);
            //rng.shuffle(ind)
            //ind = ind[:200]
            //X_diabetes, y_diabetes = X_diabetes[ind], y_diabetes[ind]
            var iris = IrisDataset.Load();
            xIris = SparseMatrix.OfMatrix(iris.Data);
            yIris = iris.Target;
        }

        /// <summary>
        /// Ridge regression convergence test using score.
        /// TODO: for this test to be robust, we should use a dataset instead of Random.
        /// </summary>
        [TestMethod]
        public void TestRidge()
        {
            var rng = new Random(0);
            const double alpha = 1.0;

            foreach (var solver in new[] {RidgeSolver.Svd, RidgeSolver.DenseCholesky, RidgeSolver.Lsqr}) 
            {
                // With more samples than features
                int nSamples = 6;
                int nFeatures = 5;
                var normal = new Normal {RandomSource = rng};
                Vector y = DenseVector.CreateRandom(nSamples, normal);
                Matrix x = DenseMatrix.CreateRandom(nSamples, nFeatures, normal);

                var ridge = new RidgeRegression(alpha: alpha, solver: solver);
                ridge.Fit(x, y);
                Assert.AreEqual(ridge.Coef.Row(0).Count, x.ColumnCount);
                Assert.IsTrue(ridge.Score(x, y) > 0.47);

                ridge.Fit(x, y, sampleWeight: DenseVector.Create(nSamples, i => 1.0));
                Assert.IsTrue(ridge.Score(x, y) > 0.47);

                // With more features than samples
                nSamples = 5;
                nFeatures = 10;
                y = DenseVector.CreateRandom(nSamples, normal);
                x = DenseMatrix.CreateRandom(nSamples, nFeatures, normal);
                ridge = new RidgeRegression(alpha: alpha, solver: solver);
                ridge.Fit(x, y);
                Assert.IsTrue(ridge.Score(x, y) > 0.9);

                ridge.Fit(x, y, sampleWeight: DenseVector.Create(nSamples, i => 1.0));
                Assert.IsTrue(ridge.Score(x, y) > 0.9);
            }
        }

        [TestMethod]
        public void TestRidgeSampleWeights()
        {
            var rng = new Random(0);
            const double alpha = 1.0;

            foreach (var solver in new[]{RidgeSolver.Svd, RidgeSolver.DenseCholesky, RidgeSolver.Lsqr})
            {
                var tests = new[] {new[]{6, 5}, new[]{5, 10}};
                foreach (var test in tests)
                {
                    int nSamples = test[0];
                    int nFeatures = test[1];
                    Vector y = DenseVector.CreateRandom(nSamples, new Normal{RandomSource = rng});
                    Matrix x = DenseMatrix.CreateRandom(nSamples, nFeatures, new Normal {RandomSource = rng});
                    Vector sampleWeight = DenseVector.CreateRandom(nSamples, new ContinuousUniform(1, 2));

                    var coefs = new RidgeRegression(alpha: alpha, solver: solver, fitIntercept:false)
                        .Fit(x, y, sampleWeight).Coef.Column(0);
                    // Sample weight can be implemented via a simple rescaling
                    // for the square loss
                    var coefs2 = new RidgeRegression(alpha, solver:solver, fitIntercept:false).Fit(
                        x.MulColumnVector(sampleWeight.Sqrt()),
                        y.PointwiseMultiply(sampleWeight.Sqrt())).Coef.Column(0);

                    Assert.IsTrue(coefs.AlmostEquals(coefs2));
                }
            }
        }

        /// <summary>
        /// Test shape of coef_ and intercept_.
        /// </summary>
        [TestMethod]
        public void TestRidgeShapes()
        {
            var rng = new Random(0);
            const int nSamples = 5;
            const int nFeatures = 10;
            Matrix x = DenseMatrix.CreateRandom(nSamples, nFeatures, new Normal {RandomSource = rng});
            Vector y = DenseVector.CreateRandom(nSamples, new Normal {RandomSource = rng});

            Matrix Y = DenseMatrix.OfColumns(y.Count, 2, new[]{y, y.Add(1)});

            var ridge = new RidgeRegression();

            ridge.Fit(x, y);
            Assert.AreEqual(ridge.Coef.Shape(), Tuple.Create(1, nFeatures));
            Assert.AreEqual(ridge.Intercept.Count, 1);

            ridge.Fit(x, MatrixExtensions.ToColumnMatrix(y));
            Assert.AreEqual(ridge.Coef.Shape(), Tuple.Create(1, nFeatures));
            Assert.AreEqual(ridge.Intercept.Count, 1);

            ridge.Fit(x, Y);
            Assert.AreEqual(ridge.Coef.Shape(), Tuple.Create(2, nFeatures));
            Assert.AreEqual(ridge.Intercept.Count, 2);
        }

        /// <summary>
        /// Test intercept with multiple targets GH issue #708.
        /// </summary>
        [TestMethod]
        public void TestRidgeIntercept()
        {
            var rng = new Random(0);
            const int nSamples = 5;
            const int nFeatures = 10;
            Matrix x = DenseMatrix.CreateRandom(nSamples, nFeatures, new Normal {RandomSource = rng});
            Vector y = DenseVector.CreateRandom(nSamples, new Normal {RandomSource = rng});

            Matrix y1 = DenseMatrix.OfColumns(y.Count, 2, new[] {y, y.Add(1)});

            var ridge = new RidgeRegression();

            ridge.Fit(x, y);
            var intercept = ridge.Intercept;

            ridge.Fit(x, y1);
            Assert.AreEqual(ridge.Intercept[0], intercept[0], 1e-10);
            Assert.AreEqual(ridge.Intercept[1], intercept[0] + 1, 1e-10);
        }

        /// <summary>
        /// Test BayesianRegression ridge classifier.
        /// TODO: test also n_samples > n_features
        /// </summary>
        [TestMethod]
        public void TestToyRidgeObject()
        {
            var x = DenseMatrix.OfArray(new double[,] {{1}, {2}});
            var y = new DenseVector(new double[] {1, 2});
            var clf = new RidgeRegression(alpha: 0.0);
            clf.Fit(x, y);
            var xTest = DenseMatrix.OfArray(new double[,] {{1}, {2}, {3}, {4}});
            Assert.AreEqual((clf.Predict(xTest) -
                DenseMatrix.OfArray(new double[,] {{1}, {2}, {3}, {4}})).FrobeniusNorm(), 0.0, 1e-10);

            Assert.AreEqual(clf.Coef.RowCount, 1);

            var y1 = DenseMatrix.OfColumns(y.Count, 2, new[] {y, y});

            clf.Fit(x, y1);

            //todo: what does this test do?
        }

        /// <summary>
        /// On alpha=0., Ridge and OLS yield the same solution.
        /// </summary>
        [TestMethod]
        public void TestRidgeVsLstsq()
        {
            var random = new Random(0);
            // we need more samples than features
            const int nSamples = 5;
            const int nFeatures = 4;
            var y = DenseVector.CreateRandom(nSamples, new Normal{RandomSource = random});
            var x = DenseMatrix.CreateRandom(nSamples, nFeatures, new Normal {RandomSource = random});

            var ridge = new RidgeRegression(alpha: 0.0, fitIntercept: false);
            var ols = new LinearRegression(fitIntercept: false);

            ridge.Fit(x, y);
            ols.Fit(x, y);
            Assert.IsTrue(ridge.Coef.AlmostEquals(ols.Coef));

            ridge.Fit(x, y);
            ols.Fit(x, y);
            Assert.IsTrue(ridge.Coef.AlmostEquals(ols.Coef));
        }

        private double? TestRidgeDiabetes(Func<Matrix<double>, Matrix<double>> filter)
        {
            var ridge = new RidgeRegression(fitIntercept: false);
            ridge.Fit(filter(xDiabetes), yDiabetes);
            return Math.Round(ridge.Score(filter(xDiabetes), yDiabetes), 5);
        }

        private double? TestMultiRidgeDiabetes(Func<Matrix<double>, Matrix<double>> filter)
        {
            // simulate several responses
            Matrix y = yDiabetes.HStack(yDiabetes);
            int nFeatures = xDiabetes.ColumnCount;

            var ridge = new RidgeRegression(fitIntercept: false);
            ridge.Fit(filter(xDiabetes), y);
            Assert.AreEqual(ridge.Coef.Shape(), Tuple.Create(2, nFeatures));
            Matrix<double> yPred = ridge.Predict(filter(xDiabetes));
            ridge.Fit(filter(xDiabetes), yDiabetes);
            var yPred1 = ridge.Predict(filter(xDiabetes));
            Assert.IsTrue(yPred1.HStack(yPred1).AlmostEquals(yPred));
            return null;
        }

        public double? TestRidgeClassifiers(Func<Matrix<double>, Matrix<double>> filter)
        {
            int nClasses = yIris.Distinct().Count();
            int nFeatures = xIris.ColumnCount;

            var clf = new RidgeClassifier<int>();
            clf.Fit(filter(xIris), yIris);

            Assert.AreEqual(clf.Coef.Shape(), Tuple.Create(nClasses, nFeatures));
            int[] yPred = clf.Predict(filter(xIris));
            var result = yIris.Zip(yPred, Tuple.Create).Sum(t => t.Item1 == t.Item2 ? 1.0 : 0.0)/yPred.Length;
            Assert.IsTrue(result >= .79);

            return null;
        }

        private double? TestTolerance(Func<Matrix<double>, Matrix<double>> filter)
        {
            var ridge = new RidgeRegression(tol: 1e-5);
            ridge.Fit(filter(xDiabetes), yDiabetes);
            double score = ridge.Score(filter(xDiabetes), yDiabetes);

            var ridge2 = new RidgeRegression(tol: 1e-3);
            ridge2.Fit(filter(xDiabetes), yDiabetes);
            double score2 = ridge2.Score(filter(xDiabetes), yDiabetes);

            Assert.IsTrue(score >= score2);

            return null;
        }

        [TestMethod]
        public void  TestDenseSparse()
        {
            foreach (Func<Func<Matrix<double>, Matrix<double>>, double?> testFunc in
                new Func<Func<Matrix<double>, Matrix<double>>, double?>[] {
                              TestRidgeDiabetes,
                              TestMultiRidgeDiabetes,
                              TestRidgeClassifiers,
                              TestTolerance})
            {
                Func<Matrix<double>, Matrix<double>> DENSE_FILTER =  DenseMatrix.OfMatrix;
                Func<Matrix<double>, Matrix<double>> SPARSE_FILTER = SparseMatrix.OfMatrix;

                // test dense matrix
                var retDense = testFunc(DENSE_FILTER);
                // test sparse matrix
                var retSparse = testFunc(SPARSE_FILTER);
                // test that the outputs are the same
                if (retDense != null && retSparse != null)
                {
                    Assert.AreEqual(retDense.Value, retSparse.Value, 1e-3);
                }
            }
        }

        /// <summary>
        /// Test class weights.
        /// </summary>
        public void TestClassWeights()
        {
            Matrix x = DenseMatrix.OfArray(new [,]
                                           {
                                               {-1.0, -1.0}, {-1.0, 0}, {-.8, -1.0},
                                               {1.0, 1.0}, {1.0, 0.0}
                                           });
            var y = new [] {1, 1, 1, -1, -1};

            var clf = new RidgeClassifier<int>(classWeightEstimator : null);
            clf.Fit(x, y);
            Assert.AreEqual(
                clf.Predict(DenseMatrix.OfArray(new[,] {{0.2, -1.0}})),
                new DenseMatrix(1, 1, new double[] {1}));

            // we give a small weights to class 1
            clf = new RidgeClassifier<int>(
                classWeightEstimator: ClassWeightEstimator<int>.Explicit(new Dictionary<int, double> {{1, 0.001}}));

            clf.Fit(x, y);

            // now the hyperplane should rotate clock-wise and
            // the prediction on this point should shift
            Assert.AreEqual(
                clf.Predict(DenseMatrix.OfArray(new [,] {{0.2, -1.0}})),
                new DenseMatrix(1, 1, new double[] {-1}));
        }
    }
}
