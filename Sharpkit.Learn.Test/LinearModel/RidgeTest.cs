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
    using Sharpkit.Learn.Datasets;
    using Sharpkit.Learn.LinearModel;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class RidgeTest
    {
        private Matrix X_diabetes;
        private Matrix y_diabetes;

        private Matrix X_iris;
        private int[] y_iris;

        [TestInitialize]
        public void TestInitialize()
        {
            var diabetes = DiabetesDataset.Load();
            X_diabetes = (Matrix)diabetes.Data.SubMatrix(0, 200, 0, diabetes.Data.ColumnCount);
            y_diabetes = ((Vector)diabetes.Target.SubVector(0, 200)).ToDenseMatrix();
            //ind = np.arange(X_diabetes.shape[0])
            //Random rng = new Random(0);
            //rng.shuffle(ind)
            //ind = ind[:200]
            //X_diabetes, y_diabetes = X_diabetes[ind], y_diabetes[ind]
            var iris = IrisDataset.Load();
            X_iris = SparseMatrix.OfMatrix(iris.Data);
            y_iris = iris.Target;
        }

        /*"""Ridge regression convergence test using score
         *             //TODO: for this test to be robust, we should use a dataset instead of np.random.
         *             */
        [TestMethod]
        public void test_ridge()
        {
            Random rng = new Random(0);
            double alpha = 1.0;

            foreach (var solver in new[] {RidgeSolver.Svd, RidgeSolver.DenseCholesky, RidgeSolver.Lsqr}) 
            {
                // With more samples than features
                int n_samples = 6;
                int n_features = 5;
                var normal = new Normal {RandomSource = rng};
                Vector y = DenseVector.CreateRandom(n_samples, normal);
                Matrix X = DenseMatrix.CreateRandom(n_samples, n_features, normal);

                var ridge = new RidgeRegression(alpha: alpha, solver: solver);
                ridge.Fit(X, y);
                Assert.AreEqual(ridge.Coef.Count, X.ColumnCount);
                Assert.IsTrue(ridge.Score(X, y) > 0.47);

                ridge.Fit(X, y, sampleWeight: DenseVector.Create(n_samples, i => 1.0));
                Assert.IsTrue(ridge.Score(X, y) > 0.47);

                // With more features than samples
                n_samples = 5;
                n_features = 10;
                y = DenseVector.CreateRandom(n_samples, normal);
                X = DenseMatrix.CreateRandom(n_samples, n_features, normal);
                ridge = new RidgeRegression(alpha: alpha, solver: solver);
                ridge.Fit(X, y);
                Assert.IsTrue(ridge.Score(X, y) > 0.9);

                ridge.Fit(X, y, sampleWeight: DenseVector.Create(n_samples, i => 1.0));
                Assert.IsTrue(ridge.Score(X, y) > 0.9);
            }
        }

        [TestMethod]
        public void test_ridge_sample_weights()
        {
            Random rng = new Random(0);
            double alpha = 1.0;

            foreach (var solver in new[]{RidgeSolver.Svd, RidgeSolver.DenseCholesky, RidgeSolver.Lsqr})
            {
                var tests = new[] {new[]{6, 5}, new[]{5, 10}};
                foreach (var test in tests)
                {
                    int n_samples = test[0];
                    int n_features = test[1];
                    Vector y = DenseVector.CreateRandom(n_samples, new Normal{RandomSource = rng});
                    Matrix X = DenseMatrix.CreateRandom(n_samples, n_features, new Normal {RandomSource = rng});
                    Vector sample_weight = DenseVector.CreateRandom(n_samples, new ContinuousUniform(1, 2));

                    var coefs = new RidgeRegression(alpha: alpha, solver: solver, fitIntercept:false).Fit(X, y, sample_weight).Coef;
                    // Sample weight can be implemented via a simple rescaling
                    // for the square loss
                    var coefs2 = new RidgeRegression(alpha, solver:solver, fitIntercept:false).Fit(
                        X.MulColumnVector(sample_weight.Sqrt()),
                        (Vector)y.PointwiseMultiply(sample_weight.Sqrt())).Coef;

                    Assert.IsTrue(coefs.AlmostEquals(coefs2));
                }
            }
        }

        /*
         * """Test shape of coef_ and intercept_
            """
         * */
        [TestMethod]
        public void test_ridge_shapes()
        {
            Random rng = new Random(0);
            int n_samples = 5;
            int n_features = 10;
            Matrix X = DenseMatrix.CreateRandom(n_samples, n_features, new Normal {RandomSource = rng});
            Vector y = DenseVector.CreateRandom(n_samples, new Normal {RandomSource = rng});

            Matrix Y = DenseMatrix.OfColumns(y.Count, 2, new[]{y, (Vector)y.Add(1)});

            var ridge = new RidgeRegression();

            ridge.Fit(X, y);
            Assert.AreEqual(ridge.CoefMatrix.Shape(), Tuple.Create(n_features, 1));
            Assert.AreEqual(ridge.InterceptVector.Count, 1);

            ridge.Fit(X, y.ToDenseMatrix());
            Assert.AreEqual(ridge.CoefMatrix.Shape(), Tuple.Create(n_features, 1));
            Assert.AreEqual(ridge.InterceptVector.Count, 1);

            ridge.Fit(X, Y);
            Assert.AreEqual(ridge.CoefMatrix.Shape(), Tuple.Create(n_features, 2));
            Assert.AreEqual(ridge.InterceptVector.Count, 2);
        }

        /*
         *     """Test intercept with multiple targets GH issue #708
    """
*/
        [TestMethod]
        public void test_ridge_intercept()
        {
            Random rng = new Random(0);
            int n_samples = 5;
            int n_features = 10;
            Matrix X = DenseMatrix.CreateRandom(n_samples, n_features, new Normal() {RandomSource = rng});
            Vector y = DenseVector.CreateRandom(n_samples, new Normal {RandomSource = rng});

            Matrix Y = DenseMatrix.OfColumns(y.Count, 2, new[] {y, y.Add(1)});

            var ridge = new RidgeRegression();

            ridge.Fit(X, y);
            var intercept = ridge.InterceptVector;

            ridge.Fit(X, Y);
            Assert.AreEqual(ridge.InterceptVector[0], intercept[0], 1e-10);
            Assert.AreEqual(ridge.InterceptVector[1], intercept[0] + 1, 1e-10);
        }

        /*
         *     """Test BayesianRegression ridge classifier

    TODO: test also n_samples > n_features
    """
         */
        [TestMethod]
        public void test_toy_ridge_object()
        {
            var X = DenseMatrix.OfArray(new double[,] {{1}, {2}});
            var Y = new DenseVector(new double[] {1, 2});
            var clf = new RidgeRegression(alpha: 0.0);
            clf.Fit(X, Y);
            var X_test = DenseMatrix.OfArray(new double[,] {{1}, {2}, {3}, {4}});
            Assert.AreEqual((clf.Predict(X_test) - DenseMatrix.OfArray(new double[,] {{1}, {2}, {3}, {4}})).FrobeniusNorm(), 0.0, 1e-10);

            Assert.AreEqual(clf.CoefMatrix.ColumnCount, 1);

            var Y1 = (Matrix)DenseMatrix.OfColumns(Y.Count, 2, new[] {Y, Y});

            clf.Fit(X, Y1);

            //todo: what does this test do?
        }

        /*
          """On alpha=0., Ridge and OLS yield the same solution."""
         */
        [TestMethod]
        public void test_ridge_vs_lstsq()
        {
            var random = new Random(0);
            // we need more samples than features
            int n_samples = 5;
            int n_features = 4;
            var y = DenseVector.CreateRandom(n_samples, new Normal{RandomSource = random});
            var X = DenseMatrix.CreateRandom(n_samples, n_features, new Normal {RandomSource = random});

            var ridge = new RidgeRegression(alpha: 0.0, fitIntercept: false);
            var ols = new LinearRegression(fitIntercept: false);

            ridge.Fit(X, y);
            ols.Fit(X, y);
            Assert.AreEqual((ridge.CoefMatrix - ols.CoefMatrix).FrobeniusNorm(), 0.0, 1e-10);

            ridge.Fit(X, y);
            ols.Fit(X, y);
            Assert.AreEqual((ridge.CoefMatrix - ols.CoefMatrix).FrobeniusNorm(), 0.0, 1e-10);
        }

        private double? test_ridge_diabetes(Func<Matrix, Matrix> filter)
        {
            var ridge = new RidgeRegression(fitIntercept: false);
            ridge.Fit(filter(X_diabetes), y_diabetes);
            return Math.Round(ridge.Score(filter(X_diabetes), y_diabetes), 5);
        }

        private double? test_multi_ridge_diabetes(Func<Matrix, Matrix> filter)
        {
            // simulate several responses
            Matrix Y = (Matrix)y_diabetes.HStack(y_diabetes);
            int n_features = X_diabetes.ColumnCount;

            var ridge = new RidgeRegression(fitIntercept: false);
            ridge.Fit(filter(X_diabetes), Y);
            Assert.AreEqual(ridge.CoefMatrix.Shape(), Tuple.Create(n_features, 2));
            Matrix Y_pred = ridge.Predict(filter(X_diabetes));
            ridge.Fit(filter(X_diabetes), y_diabetes);
            var y_pred = ridge.Predict(filter(X_diabetes));
            Assert.IsTrue(y_pred.HStack(y_pred).AlmostEquals(Y_pred));
            return null;
        }

        public double? test_ridge_classifiers(Func<Matrix, Matrix> filter)
        {
            int n_classes = y_iris.Distinct().Count();
            int n_features = X_iris.ColumnCount;

            var clf = new RidgeClassifier<int>();
            clf.Fit(filter(X_iris), y_iris);
            //TODO: CoefMatrix is transposed in scikit
            Assert.AreEqual(clf.CoefMatrix.Shape(), Tuple.Create(n_features, n_classes));
            int[] y_pred = clf.Predict(filter(X_iris));
            var result = y_iris.Zip(y_pred, Tuple.Create).Sum(t => t.Item1 == t.Item2 ? 1.0 : 0.0)/y_pred.Length;
            Assert.IsTrue(result >= .79);

            return null;
        }

        private double? test_tolerance(Func<Matrix, Matrix> filter)
        {
            var ridge = new RidgeRegression(tol: 1e-5);
            ridge.Fit(filter(X_diabetes), y_diabetes);
            double score = ridge.Score(filter(X_diabetes), y_diabetes);

            var ridge2 = new RidgeRegression(tol: 1e-3);
            ridge2.Fit(filter(X_diabetes), y_diabetes);
            double score2 = ridge2.Score(filter(X_diabetes), y_diabetes);

            Assert.IsTrue(score >= score2);

            return null;
        }

        [TestMethod]
        public void  test_dense_sparse()
        {
            foreach (Func<Func<Matrix, Matrix>, double?>  test_func in new Func<Func<Matrix, Matrix>, double?>[] {
                              test_ridge_diabetes,
                              test_multi_ridge_diabetes,
                              test_ridge_classifiers,
                              test_tolerance})
            {
                Func<Matrix, Matrix> DENSE_FILTER =  X => DenseMatrix.OfMatrix(X);
                Func<Matrix, Matrix> SPARSE_FILTER = X => SparseMatrix.OfMatrix(X);

                // test dense matrix
                var ret_dense = test_func(DENSE_FILTER);
                // test sparse matrix
                var ret_sparse = test_func(SPARSE_FILTER);
                // test that the outputs are the same
                if (ret_dense != null && ret_sparse != null)
                {
                    Assert.AreEqual(ret_dense.Value, ret_sparse.Value, 1e-3);
                }
            }
        }

        /*
         * """
    Test class weights.
    """*/
        public void test_class_weights()
        {
            Matrix X = DenseMatrix.OfArray(new [,]
                                           {
                                               {-1.0, -1.0}, {-1.0, 0}, {-.8, -1.0},
                                               {1.0, 1.0}, {1.0, 0.0}
                                           });
            var y = new int[] {1, 1, 1, -1, -1};

            var clf = new RidgeClassifier<int>(classWeight : null);
            clf.Fit(X, y);
            Assert.AreEqual(
                clf.Predict(DenseMatrix.OfArray(new[,] {{0.2, -1.0}})),
                new DenseMatrix(1, 1, new double[] {1}));

            // we give a small weights to class 1
            clf = new RidgeClassifier<int>(classWeight: ClassWeight<int>.Explicit(new Dictionary<int, double> {{1, 0.001}}));
            clf.Fit(X, y);

            // now the hyperplane should rotate clock-wise and
            // the prediction on this point should shift
            Assert.AreEqual(
                clf.Predict(DenseMatrix.OfArray(new [,] {{0.2, -1.0}})),
                new DenseMatrix(1, 1, new double[] {-1}));
        }
    }
}
