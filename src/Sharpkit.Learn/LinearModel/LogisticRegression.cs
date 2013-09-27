// -----------------------------------------------------------------------
// <copyright file="LogisticRegression.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.LinearModel
{
    using System;
    using MathNet.Numerics.LinearAlgebra.Double;
    using MathNet.Numerics.LinearAlgebra.Generic;

    /// <summary>
    /// <para>
    /// Logistic Regression (aka logit, MaxEnt) classifier.
    /// </para>
    /// <para>
    /// In the multiclass case, the training algorithm uses a one-vs.-all (OvA)
    /// scheme, rather than the "true" multinomial LR.
    /// </para>
    /// <para>
    /// This class implements L1 and L2 regularized logistic regression using the
    /// `liblinear` library. It can handle both dense and sparse input.
    /// </para>
    /// </summary>
    /// <remarks>
    /// <para>
    /// The underlying Liblinear implementation uses a random number generator to
    /// select features when fitting the model. It is thus not uncommon,
    /// to have slightly different results for the same input data. If
    /// that happens, try with a smaller tol parameter.
    /// </para>
    /// <para>
    /// References:
    /// </para>
    /// <para>
    /// LIBLINEAR -- A Library for Large Linear Classification
    /// http://www.csie.ntu.edu.tw/~cjlin/liblinear/
    /// </para>
    /// <para>
    /// Hsiang-Fu Yu, Fang-Lan Huang, Chih-Jen Lin (2011). Dual coordinate descent
    ///    methods for logistic regression and maximum entropy models.
    ///    Machine Learning 85(1-2):41-75.
    ///    http://www.csie.ntu.edu.tw/~cjlin/papers/maxent_dual.pdf
    /// </para>
    /// </remarks>
    public class LogisticRegression<TLabel> : LinearClassifier<TLabel> where TLabel : IEquatable<TLabel>
    {
        private readonly LibLinearBase<TLabel> libLinearBase;

        /// <summary>
        /// Gets or sets a value indicating whether formulation is primal or dual. Dual formulation is only
        /// implemented for l2 penalty. Prefer dual=false when
        /// nSamples > nFeatures.
        /// </summary>
        public bool Dual
        {
            get { return this.libLinearBase.Dual; }
        }

        public double InterceptScaling
        {
            get { return this.libLinearBase.interceptScaling; }
        }

        /// <summary>
        /// Gets or sets inverse of regularization strength; must be a positive float.
        /// Like in support vector machines, smaller values specify stronger
        /// regularization.
        /// </summary>
        public double C
        {
            get { return this.libLinearBase.C; }
        }

        /// <summary>
        /// Initializes a new instance of the logistic regression class.
        /// </summary>
        /// <param name="penalty">Used to specify the norm used in the penalization.</param>
        /// <param name="dual">
        /// Dual or primal formulation. Dual formulation is only
        /// implemented for l2 penalty. Prefer dual=false when
        /// nSamples > nFeatures.
        /// </param>
        /// <param name="tol"></param>
        /// <param name="c">Inverse of regularization strength; must be a positive float.
        /// Like in support vector machines, smaller values specify stronger
        /// regularization.
        /// </param>
        /// <param name="fitIntercept">
        /// Specifies if a constant (a.k.a. bias or intercept) should be
        /// added the decision function.
        /// </param>
        /// <param name="interceptScaling">
        /// when fitIntercept is true, instance vector x becomes
        /// [x, self.intercept_scaling],
        /// i.e. a "synthetic" feature with constant value equals to
        /// interceptScaling is appended to the instance vector.
        /// The intercept becomes intercept_scaling * synthetic feature weight
        /// Note! the synthetic feature weight is subject to l1/l2 regularization
        /// as all other features.
        /// To lessen the effect of regularization on synthetic feature weight
        /// (and therefore on the intercept) interceptScaling has to be increased
        /// </param>
        /// <param name="classWeightEstimator">Set the parameter C of class i to class_weight[i]*C for
        /// SVC. If not given or ClassWeight.Uniform is used, all classes are supposed to have
        /// weight one. ClassWeight.Auto uses the values of y to
        //  automatically adjust weights inversely proportional to
        /// class frequencies.</param>
        /// <param name="random">
        /// The seed of the pseudo random number generator to use when
        /// shuffling the data.
        /// </param>
        public LogisticRegression(
            Norm penalty = Norm.L2,
            bool dual = false,
            double tol = 1e-4,
            double c = 1.0,
            bool fitIntercept = true,
            double interceptScaling = 1,
            ClassWeightEstimator<TLabel> classWeightEstimator = null,
            Random random = null) : base(fitIntercept, classWeightEstimator)
        {
            libLinearBase = new LibLinearBase<TLabel>(
                this,
                penalty,
                Loss.LogisticRegression,
                dual,
                tol,
                c,
                interceptScaling: interceptScaling,
                random: random);
        }

        /// <summary>
        /// Fit the model according to the given training data.
        /// </summary>
        /// <param name="x">[nSamples, nFeatures]. Training vectors,
        /// where nSamples is the number of samples and nFeatures
        /// is the number of features.</param>
        /// <param name="y">[nSamples] Target class labels.</param>
        /// <returns>Reference to itself.</returns>
        public override IClassifier<TLabel> Fit(Matrix<double> x, TLabel[] y)
        {
            this.libLinearBase.Fit(x, y);
            return this;
        }

        /// <summary>
        /// Probability estimates.
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <param name="x">[n_samples, n_features]</param>
        /// <returns>[n_samples, n_classes]
        ///       Returns the probability of the sample for each class in the model,
        ///       where classes are ordered as they are in <see cref="Classes"/>.
        /// </returns>
        public override Matrix<double> PredictProba(Matrix<double> x)
        {
            return PredictProbaLr(x);
        }

        /// <summary>
        /// Log of probability estimates.
        ///
        /// The returned estimates for all classes are ordered by the
        /// label of classes.
        /// </summary>
        /// <param name="x">[n_samples, n_features].</param>
        /// <returns>[n_samples, n_classes]
        /// Returns the log-probability of the sample for each class in the
        /// model, where classes are ordered as they are in <see cref="Classes"/>.
        /// </returns>
        public override Matrix<double> PredictLogProba(Matrix<double> x)
        {
            return this.PredictProba(x).Log();
        }

        /// <summary>
        /// Gets ordered list of class labeled discovered int <see cref="LogisticRegression{TLabel}.Fit"/>.
        /// </summary>
        public override TLabel[] Classes
        {
            get { return this.libLinearBase.Classes; }
        }

        /// <summary>
        /// Convert coefficient matrix to sparse format.
        /// Converts the <see cref="LinearModel.Coef"/> member to a <see cref="SparseMatrix"/>, which for
        /// L1-regularized models can be much more memory- and storage-efficient
        /// than the usual numpy.ndarray representation.
        ///
        /// The <see cref="LinearModel.Intercept"/> member is not converted.
        /// </summary>
        /// <remarks>
        /// For non-sparse models, i.e. when there are not many zeros in <see cref="LinearModel.Coef"/>,
        /// this may actually *increase* memory usage, so use this method with
        /// care. A rule of thumb is that the number of zero elements, which can
        /// be computed with ``(Coef == 0).Sum()``, must be more than 50% for this
        /// to provide significant benefits.
        ///
        /// After calling this method, further fitting with the partial_fit
        /// method (if any) will not work until you call densify.
        /// </remarks>
        public LogisticRegression<TLabel> Sparsify()
        {
            if (this.Coef is DenseMatrix)
            {
                this.Coef = SparseMatrix.OfMatrix(this.Coef);
            }

            return this;
        }

        /// <summary>
        /// Convert coefficient matrix to dense array format.
        ///
        /// Converts the <see cref="LinearModel.Coef"/> member (back) to a numpy.ndarray. This is the
        /// default format of <see cref="LinearModel.Coef"/> and is required for fitting, so calling
        /// this method is only required on models that have previously been
        /// sparsified; otherwise, it is a no-op.
        /// </summary>
        /// <returns></returns>
        public LogisticRegression<TLabel> Densify()
        {
            if (this.Coef is SparseMatrix)
            {
                this.Coef = DenseMatrix.OfMatrix(this.Coef);
            }

            return this;
        }
    }
}
