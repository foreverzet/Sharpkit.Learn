// -----------------------------------------------------------------------
// <copyright file="SvmTest.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Test.Svm
{
    using System;
    using System.Linq;
    using System.Collections.Generic;
    using Liblinear;
    using MathNet.Numerics.LinearAlgebra.Double;
    using Microsoft.VisualStudio.TestTools.UnitTesting;
    using Sharpkit.Learn.Datasets;
    using Sharpkit.Learn.LinearModel;
    using Sharpkit.Learn.Utils;
    using Sharpkit.Learn;
    using Sharpkit.Learn.Svm;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    [TestClass]
    public class SvmTest
    {
        private DenseMatrix X = DenseMatrix.OfArray(
            new double[,]
                {
                    {-2, -1},
                    {-1, -1},
                    {-1, -2},
                    {1, 1},
                    {1, 2},
                    {2, 1}
                });

        private int[] Y = new[] {1, 1, 1, 2, 2, 2};
        
        private DenseMatrix T = DenseMatrix.OfArray(new double[,]
                                                        {
                                                            {-1, -1},
                                                            {2, 2},
                                                            {3, 2}
                                                        });
        private int[] true_result = new[] {1, 2, 2};

        private DenseMatrix X2 =
            DenseMatrix.OfArray(new double[,]
                                    {
                                        {0, 0, 0},
                                        {1, 1, 1},
                                        {2, 0, 0},
                                        {0, 0, 2},
                                        {3, 3, 3}
                                    });

        private int[] Y2 = new[] {1, 2, 2, 2, 3};
        private DenseMatrix T2 = DenseMatrix.OfArray(
            new double[,]
                {
                    {-1, -1, -1},
                    {1, 1, 1},
                    {2, 2, 2}
                });

        private int[] true_result2 = new[] {1, 2, 3};

        private IrisDataset iris = IrisDataset.Load();

        [ClassInitialize]
        public static void ClassInitialize(TestContext t)
        {
            Linear.disableDebugOutput();
        }

        /// <summary>
        /// Test parameters on classes that make use of libsvm.
        /// </summary>
        [TestMethod]
        public void TestLibsvmParameters()
        {
            var clf = new Svc<int>(kernel: Kernel.Linear);
            clf.Fit(X, Y);
            Assert.IsTrue(clf.DualCoef.AlmostEquals(DenseMatrix.OfArray(new[,] {{0.25, -.25}})));
            Assert.IsTrue(clf.Support.SequenceEqual(new[] {1, 3}));
            Assert.IsTrue(
                clf.SupportVectors.AlmostEquals(
                DenseMatrix.OfRows(2, X.ColumnCount, new[] {X.Row(1), X.Row(3)})));

            Assert.IsTrue(clf.Intercept.SequenceEqual(new[] {0.0}));
            Assert.IsTrue(clf.Predict(X).SequenceEqual(Y));
        }

        /// <summary>
        /// Check that sparse SVC gives the same result as SVC"
        /// </summary>
        [TestMethod]
        public void TestSvc()
        {
            var clf = new Svc<int>(kernel: Kernel.Linear, probability: true);
            clf.Fit(X, Y);

            var spClf = new Svc<int>(kernel: Kernel.Linear, probability: true);
            spClf.Fit(SparseMatrix.OfMatrix(X), Y);

            Assert.IsTrue(spClf.Predict(T).SequenceEqual(true_result));

            Assert.IsTrue(spClf.SupportVectors is SparseMatrix);
            Assert.IsTrue(clf.SupportVectors.AlmostEquals(spClf.SupportVectors));

            Assert.IsTrue(spClf.DualCoef is SparseMatrix);
            Assert.IsTrue(clf.DualCoef.AlmostEquals(spClf.DualCoef));

            Assert.IsTrue(spClf.Coef is SparseMatrix);
            Assert.IsTrue(clf.Coef.AlmostEquals(spClf.Coef));
            Assert.IsTrue(clf.Support.SequenceEqual(spClf.Support));
            Assert.IsTrue(clf.Predict(T).SequenceEqual(spClf.Predict(T)));

            // refit with a different dataset

            clf.Fit(X2, Y2);
            spClf.Fit(SparseMatrix.OfMatrix(X2), Y2);
            Assert.IsTrue(clf.SupportVectors.AlmostEquals(spClf.SupportVectors));
            Assert.IsTrue(clf.DualCoef.AlmostEquals(spClf.DualCoef));
            Assert.IsTrue(clf.Coef.AlmostEquals(spClf.Coef));
            Assert.IsTrue(clf.Support.SequenceEqual(spClf.Support));
            Assert.IsTrue(clf.Predict(T2).SequenceEqual(spClf.Predict(T2)));
            Assert.IsTrue(clf.PredictProba(T2).AlmostEquals(spClf.PredictProba(T2), 0.001));
        }


        /// <summary>
        /// Check consistency on dataset iris.
        /// </summary>
        [TestMethod]
        public void TestLibsvmIris()
        {
            // shuffle the dataset so that labels are not ordered
            foreach (var k in new[] {Kernel.Linear, Kernel.Rbf})
            {
                var clf = new Svc<int>(kernel: k);
                clf.Fit(iris.Data, iris.Target);
                var pred = clf.Predict(iris.Data);
                var matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
                Assert.IsTrue(1.0*matchingN/pred.Length > 0.9);
                Assert.IsTrue(clf.Classes.SequenceEqual(clf.Classes.OrderBy(v => v)));
            }
        }

        /// <summary>
        /// Test whether SVCs work on a single sample given as a 1-d array.
        /// </summary>
        [TestMethod]
        public void TestSingleSample_1D()
        {
            var clf = new Svc<int>();
            clf.Fit(X, Y);
            var p = clf.Predict(X.Row(0).ToRowMatrix());
            Assert.AreEqual(Y[0], p[0]);

            //todo:
            //clf = svm.LinearSVC(random_state=0).fit(X, Y)
            //clf.predict(X[0])
        }

        /// <summary>
        /// SVC with a precomputed kernel.
        ///
        /// We test it with a toy dataset and with iris.
        /// </summary>
        [TestMethod]
        public void TestPrecomputed()
        {
            var clf = new Svc<int>(kernel: Kernel.Precomputed);
            // Gram matrix for train data (square matrix)
            // (we use just a linear kernel)
            var k = X*(X.Transpose());
            clf.Fit(k, Y);
            // Gram matrix for test data (rectangular matrix)
            var kt = T*X.Transpose();
            var pred = clf.Predict(kt);
            try
            {
                clf.Predict(kt.Transpose());
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            Assert.IsTrue(clf.DualCoef.AlmostEquals(DenseMatrix.OfArray(new[,] {{0.25, -.25}})));
            Assert.IsTrue(clf.Support.SequenceEqual(new[] {1, 3}));
            Assert.IsTrue(clf.Intercept.SequenceEqual(new[] {0.0}));
            Assert.IsTrue(pred.SequenceEqual(true_result));

            // Gram matrix for test data but compute KT[i,j]
            // for support vectors j only.
            kt = kt.CreateMatrix(kt.RowCount, kt.ColumnCount);
            for (int i = 0; i < T.RowCount; i++)
            {
                foreach (var j in clf.Support)
                {
                    kt[i, j] = T.Row(i)*X.Row(j);
                }
            }

            pred = clf.Predict(kt);
            Assert.IsTrue(pred.SequenceEqual(true_result));

            // same as before, but using a callable function instead of the kernel
            // matrix. kernel is just a linear kernel

            clf = new Svc<int>(kernel: Kernel.FromFunction((x, y) => x*y.Transpose()));
            clf.Fit(X, Y);
            pred = clf.Predict(T);

            Assert.IsTrue(clf.DualCoef.AlmostEquals(DenseMatrix.OfArray(new[,] {{0.25, -.25}})));
            Assert.IsTrue(clf.Support.SequenceEqual(new[] {1, 3}));
            Assert.IsTrue(clf.Intercept.SequenceEqual(new[] {0.0}));
            Assert.IsTrue(pred.SequenceEqual(true_result));

            // test a precomputed kernel with the iris dataset
            // and check parameters against a linear SVC
            clf = new Svc<int>(kernel: Kernel.Precomputed);
            var clf2 = new Svc<int>(kernel: Kernel.Linear);
            k = iris.Data*iris.Data.Transpose();
            clf.Fit(k, iris.Target);
            clf2.Fit(iris.Data, iris.Target);
            pred = clf.Predict(k);
            Assert.IsTrue(clf.Support.SequenceEqual(clf2.Support));
            Assert.IsTrue(clf.DualCoef.AlmostEquals(clf2.DualCoef));
            Assert.IsTrue(clf.Intercept.AlmostEquals(clf2.Intercept));

            var matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0*matchingN/pred.Length > 0.99);

            // Gram matrix for test data but compute KT[i,j]
            // for support vectors j only.
            k = k.CreateMatrix(k.RowCount, k.ColumnCount);
            for (int i = 0; i < iris.Data.RowCount; i++)
            {
                foreach (var j in clf.Support)
                    k[i, j] = iris.Data.Row(i)*iris.Data.Row(j);
            }

            pred = clf.Predict(k);
            matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0*matchingN/pred.Length > 0.99);

            clf = new Svc<int>(kernel: Kernel.FromFunction((x, y) => x*y.Transpose()));
            clf.Fit(iris.Data, iris.Target);
            matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
            Assert.IsTrue(1.0*matchingN/pred.Length > 0.99);
        }

        /// <summary>
        /// Make sure some tweaking of parameters works.
        /// 
        /// We change clf.dual_coef_ at run time and expect .predict() to change
        /// accordingly. Notice that this is not trivial since it involves a lot
        /// of C/Python copying in the libsvm bindings.
        ///
        /// The success of this test ensures that the mapping between libsvm and
        /// the python classifier is complete.
        /// </summary>
        public void TestTweakParams()
        {
            var clf = new Svc<int>(kernel: Kernel.Linear, c: 1.0);
            clf.Fit(X, Y);
            Assert.IsTrue(clf.DualCoef.AlmostEquals(DenseMatrix.OfArray(new[,]{{0.25, -0.25}})));
            Assert.IsTrue(clf.Predict(DenseMatrix.OfArray(new[,] {{-.1, -.1}})).SequenceEqual(new[] {1}));
            clf.DualCoef = DenseMatrix.OfArray(new[,] {{0.0, 1.0}});
            Assert.IsTrue(clf.Predict(DenseMatrix.OfArray(new[,] {{-.1, -.1}})).SequenceEqual(new[] {2}));
        }

        /// <summary>
        /// Predict probabilities using SVC
        ///
        /// This uses cross validation, so we use a slightly bigger testing set.
        /// </summary>
        [TestMethod]
        public void TestProbability()
        {
            foreach (var clf in new IClassifier<int>[]
                                    {
                                        new Svc<int>(probability: true, c: 1.0),
                                        //svm.NuSVC(probability=True)
                                    }
                )
            {
                clf.Fit(iris.Data, iris.Target);

                var probPredict = clf.PredictProba(iris.Data);
                Assert.IsTrue(
                    probPredict.SumOfEveryRow().AlmostEquals(
                        DenseVector.OfEnumerable(Enumerable.Repeat(1.0, iris.Data.RowCount))));

                var pred = probPredict.RowEnumerator().Select(r => r.Item2.MaximumIndex()).ToArray();
                var matchingN = pred.Zip(iris.Target, Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
                Assert.IsTrue(1.0*matchingN/pred.Length > 0.9);
                Assert.IsTrue(clf.PredictProba(iris.Data).AlmostEquals(clf.PredictLogProba(iris.Data).Exp()));
            }
        }

        /// <summary>
        /// Test decision_function
        ///
        ///    Sanity check, test that decision_function implemented in python
        ///    returns the same as the one in libsvm
        /// </summary>
        [TestMethod]
        public void TestDecisionFunction()
        {
            // multi class:
            var clf = new Svc<int>(kernel: Kernel.Linear, c: 0.1);
            clf.Fit(iris.Data, iris.Target);

            var dec = (iris.Data*clf.Coef.Transpose()).AddRowVector(clf.Intercept);

            Assert.IsTrue(dec.AlmostEquals(clf.DecisionFunction(iris.Data)));

            // binary:
            clf.Fit(X, Y);
            dec = (X*clf.Coef.Transpose()).AddRowVector(clf.Intercept);
            int[] prediction = clf.Predict(X);
            Assert.IsTrue(dec.AlmostEquals(clf.DecisionFunction(X)));

            var b = clf.DecisionFunction(X).Column(0).Select(v => clf.Classes[v > 0 ? 1 : 0]);
            Assert.IsTrue(prediction.SequenceEqual(b));

            var expected = DenseMatrix.OfArray(new[,] {{-1.0}, {-0.66}, {-1.0}, {0.66}, {1.0}, {1.0}});
            Assert.IsTrue(clf.DecisionFunction(X).AlmostEquals(expected, 1E-2));
        }

        /// <summary>
        ///     Test class weights
        /// </summary>
        /// <returns></returns>
        [TestMethod]
        public void test_weight()
        {
            var clf = new Svc<int>(classWeightEstimator: ClassWeightEstimator<int>.Explicit(new Dictionary<int, double> {{1, 0.1}}));

            // we give a small weights to class 1
            clf.Fit(X, Y);
            // so all predicted values belong to class 2
            Assert.IsTrue(clf.Predict(X).SequenceEqual(Enumerable.Repeat(2, 6)));
            /*
    X_, y_ = make_classification(n_samples=200, n_features=10,
                                 weights=[0.833, 0.167], random_state=2)

    for clf in (linear_model.LogisticRegression(),
                svm.LinearSVC(random_state=0), svm.SVC()):
        clf.set_params(class_weight={0: .1, 1: 10})
        clf.fit(X_[:100], y_[:100])
        y_pred = clf.predict(X_[100:])
        assert_true(f1_score(y_[100:], y_pred) > .3)
             * */
        }

        /// <summary>
        /// Test class weights for imbalanced data
        /// </summary>
        [TestMethod]
        public void TestAutoWeight()
        {
            // We take as dataset the two-dimensional projection of iris so
            // that it is not separable and remove half of predictors from
            // class 1.
            // We add one to the targets as a non-regression test: class_weight="auto"
            // used to work only when the labels where a range [0..K).
            var x = iris.Data.SubMatrix(0, iris.Data.RowCount, 0, 2);
            var y = iris.Target.Select(v => v + 1).ToArray();

            var indToDelete = y.Indices(v => v > 2).Where((v, i) => i%2 == 0).ToList();
            var unbalancedIndices = Enumerable.Range(0, y.Length).Where(v => !indToDelete.Contains(v)).ToList();

            int[] yInd;
            int[] classes = Fixes.Unique(y.ElementsAt(unbalancedIndices), out yInd);

            var classWeights = ClassWeightEstimator<int>.Auto.ComputeWeights(classes, yInd);
            Assert.AreEqual(2, classWeights.MaximumIndex());

            foreach (var clf in new IClassifier<int>[]
                                    {
                                        new Svc<int>(kernel: Kernel.Linear, classWeightEstimator:ClassWeightEstimator<int>.Auto),
                                        //svm.LinearSVC(random_state=0),
                                        new LogisticRegression<int>(classWeightEstimator : ClassWeightEstimator<int>.Auto)
                                    })
            {
                // check that score is better when class='auto' is set.
                clf.Fit(x.RowsAt(unbalancedIndices), y.ElementsAt(unbalancedIndices));
                var y_pred = clf.Predict(x);
                clf.Fit(x.RowsAt(unbalancedIndices), y.ElementsAt(unbalancedIndices));
                var yPredBalanced = clf.Predict(x);
                var a = Learn.Metrics.Metrics.F1Score(y, y_pred);
                var b = Learn.Metrics.Metrics.F1Score(y, yPredBalanced);
                Assert.IsTrue(a.Zip(b, Tuple.Create).All(t => t.Item1 <= t.Item2));
            }
        }

        [TestMethod]
        public void TestSvcWithCallableKernel()
        {
            // create SVM with callable linear kernel, check that results are the same
            // as with built-in linear kernel
            var svmCallable = new Svc<int>(kernel: Kernel.FromFunction((x, y) => x*(y.Transpose())),
                                            probability: true);

            svmCallable.Fit(X, Y);
            var svmBuiltin = new Svc<int>(kernel: Kernel.Linear, probability: true);
            svmBuiltin.Fit(X, Y);

            Assert.IsTrue(svmCallable.DualCoef.AlmostEquals(svmBuiltin.DualCoef));
            Assert.IsTrue(svmCallable.Intercept.AlmostEquals(svmBuiltin.Intercept));
            Assert.IsTrue(svmCallable.Predict(X).SequenceEqual(svmBuiltin.Predict(X)));

            Assert.IsTrue(svmCallable.PredictProba(X).AlmostEquals(svmBuiltin.PredictProba(X), 1));
            Assert.IsTrue(svmCallable.DecisionFunction(X).AlmostEquals(svmBuiltin.DecisionFunction(X), 2));
        }

        [TestMethod]
        [ExpectedException(typeof (ArgumentException))]
        public void TestSvcBadKernel()
        {
            var svc = new Svc<int>(kernel: Kernel.FromFunction((x, y) => x));
            svc.Fit(X, Y);
        }

        [TestMethod]
        public void TestSvcWithCustomKernel()
        {
            var clfLin = new Svc<int>(kernel: Kernel.Linear);
            clfLin.Fit(SparseMatrix.OfMatrix(X), Y);
            var clfMylin =
                new Svc<int>(kernel: Kernel.FromFunction((x, y) => x*y.Transpose()));
            clfMylin.Fit(SparseMatrix.OfMatrix(X), Y);
            Assert.IsTrue(
                clfLin.Predict(SparseMatrix.OfMatrix(X)).SequenceEqual(clfMylin.Predict(SparseMatrix.OfMatrix(X))));
        }

        /// <summary>
        /// Test the sparse SVC with the iris dataset;
        /// </summary>
        [TestMethod]
        public void TestSvcIris()
        {
            foreach (var k in new[] {Kernel.Linear, Kernel.Poly, Kernel.Rbf})
            {
                var spClf = new Svc<int>(kernel: k);
                spClf.Fit(SparseMatrix.OfMatrix(iris.Data), iris.Target);
                var clf = new Svc<int>(kernel: k);
                clf.Fit(DenseMatrix.OfMatrix(iris.Data), iris.Target);

                Assert.IsTrue(clf.SupportVectors.AlmostEquals(spClf.SupportVectors));
                Assert.IsTrue(clf.DualCoef.AlmostEquals(spClf.DualCoef));
                Assert.IsTrue(
                    clf.Predict(DenseMatrix.OfMatrix(iris.Data)).SequenceEqual(
                        spClf.Predict(SparseMatrix.OfMatrix(iris.Data))));

                if (k == Kernel.Linear)
                {
                    Assert.IsTrue(clf.Coef.AlmostEquals(spClf.Coef));
                }
            }
        }

        /*
        /// <summary>
        /// Test that it gives proper exception on deficient input
        /// </summary>
        [TestMethod]
        public void TestError()
        {
            // impossible value of C
            try
            {
                new Svc<int>(c: -1).Fit(X, Y);
                Assert.Fail();
            }
            catch (ArgumentException)
            {
            }

            // impossible value of nu
            //clf = svm.NuSVC(nu=0.0)
            //assert_raises(ValueError, clf.fit, X_sp, Y)

            //Y2 = Y[:-1]  # wrong dimensions for labels
            //assert_raises(ValueError, clf.fit, X_sp, Y2)

            var clf = new Svc<int>();
            clf.Fit(SparseMatrix.OfMatrix(X), Y);
            Assert.IsTrue(clf.Predict(T).SequenceEqual(true_result));
        }*/

        /// <summary>
        /// Test class weights
        /// </summary>
        [TestMethod]
        [Ignore]
        public void TestWeight()
        {
            var classification =
                SampleGenerator.MakeClassification(
                    nSamples: 200,
                    nFeatures: 100,
                    weights: new[] {0.833, 0.167}.ToList(),
                    randomState: new Random(0));

            var classWeight = ClassWeightEstimator<int>.Explicit(new Dictionary<int, double> {{0, 5}});
            Matrix x = SparseMatrix.OfMatrix(classification.X);
            foreach (var clf in new IClassifier<int>[]
                                    {
                                        new LogisticRegression<int>(classWeightEstimator: classWeight),
                                        //new LinearSvc(classWeight:classWeight, random_state=0),
                                        //new Svc<int>(classWeight: classWeight)
                                    })
            {
                clf.Fit(x.SubMatrix(0, 180, 0, x.ColumnCount), classification.Y.Take(180).ToArray());
                var yPred = clf.Predict(x.SubMatrix(180, x.RowCount - 180, 0, x.ColumnCount));

                var matchingN =
                    yPred.Zip(classification.Y.Skip(180), Tuple.Create).Where(t => t.Item1 == t.Item2).Count();
                Assert.IsTrue(matchingN >= 11);
            }
        }

        /// <summary>
        /// Test on a subset from the 20newsgroups dataset.
        ///
        /// This catchs some bugs if input is not correctly converted into
        /// sparse format or weights are not correctly initialized.
        /// </summary>
        [TestMethod]
        public void TestSparseRealdata()
        {
            var x = new SparseMatrix(80, 36);
            x[7, 6] = 0.03771744;
            x[39, 5] = 0.1003567;
            x[77, 35] = 0.01174647;
            x[77, 31] = 0.027069;

            var y = new[]
                             {
                                 1.0, 0.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, 2.0,
                                 0.0, 2.0, 0.0, 3.0, 0.0, 3.0, 0.0, 1.0, 1.0, 3.0, 2.0, 3.0, 2.0,
                                 0.0, 3.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 3.0, 1.0,
                                 3.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0,
                                 0.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0,
                                 3.0, 0.0, 0.0, 2.0, 2.0, 1.0, 3.0, 1.0, 1.0, 0.0, 1.0, 2.0, 1.0,
                                 1.0, 3.0
                             };

            var clf = new Svc<double>(kernel: Kernel.Linear);
            clf.Fit(DenseMatrix.OfMatrix(x), y);
            var spClf = new Svc<double>(kernel: Kernel.Linear);
            spClf.Fit(x, y);

            Assert.IsTrue(clf.SupportVectors.AlmostEquals(spClf.SupportVectors));
            Assert.IsTrue(clf.DualCoef.AlmostEquals(spClf.DualCoef));
        }
    }
}
