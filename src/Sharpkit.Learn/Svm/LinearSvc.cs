// -----------------------------------------------------------------------
// <copyright file="LinearSvc.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using MathNet.Numerics.LinearAlgebra.Generic;

namespace Sharpkit.Learn.Svm
{
    using System;
    using Sharpkit.Learn.LinearModel;

    /// <summary>
    /// Linear Support Vector Classification.
    ///
    /// Similar to SVC with parameter kernel='linear', but implemented in terms of
    /// liblinear rather than libsvm, so it has more flexibility in the choice of
    /// penalties and loss functions and should scale better (to large numbers of
    /// samples).
    ///
    /// This class supports both dense and sparse input and the multiclass support
    /// is handled according to a one-vs-the-rest scheme.
    /// </summary>
    /// <remarks>
    /// The underlying C implementation uses a random number generator to
    /// select features when fitting the model. It is thus not uncommon,
    /// to have slightly different results for the same input data. If
    /// that happens, try with a smaller tol parameter.
    /// 
    /// The underlying implementation (liblinear) uses a sparse internal
    /// representation for the data that will incur a memory copy.
    /// 
    /// **References:**
    /// `LIBLINEAR: A Library for Large Linear Classification
    /// <http://www.csie.ntu.edu.tw/~cjlin/liblinear/>`__
    ///
    /// See also
    /// --------
    /// SVC
    /// Implementation of Support Vector Machine classifier using libsvm:
    /// the kernel can be non-linear but its SMO algorithm does not
    /// scale to large number of samples as LinearSVC does.
    ///
    /// Furthermore SVC multi-class mode is implemented using one
    /// vs one scheme while LinearSVC uses one vs the rest. It is
    /// possible to implement one vs the rest with SVC by using the
    ///    :class:`sklearn.multiclass.OneVsRestClassifier` wrapper.
    ///
    ///    Finally SVC can fit dense data without memory copy if the input
    ///    is C-contiguous. Sparse data will still incur memory copy though.
    ///
    /// sklearn.linear_model.SGDClassifier
    ///    SGDClassifier can optimize the same cost function as LinearSVC
    ///    by adjusting the penalty and loss parameters. Furthermore
    ///    SGDClassifier is scalable to large number of samples as it uses
    ///    a Stochastic Gradient Descent optimizer.
    ///
    ///    Finally SGDClassifier can fit both dense and sparse data without
    ///    memory copy if the input is C-contiguous or CSR.
    /// </remarks>
    public class LinearSvc<TLabel> : LinearClassifier<TLabel> where TLabel : IEquatable<TLabel>
    {
        private LibLinearBase<TLabel> libLinearBase;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="c">Penalty parameter C of the error term.</param>
        /// <param name="loss">Loss function.</param>
        /// <param name="penalty">Specifies the norm used in the penalization. The 'l2'
        /// penalty is the standard used in SVC. The 'l1' leads to `coef_`
        /// vectors that are sparse.</param>
        /// <param name="dual">Select the algorithm to either solve the dual or primal
        /// optimization problem. Prefer dual=False when n_samples > n_features.</param>
        /// <param name="tol">Tolerance for stopping criteria.</param>
        /// <param name="multiclass">Determines the multi-class strategy if `y` contains more than
        /// two classes. If `crammer_singer` is chosen, the options loss, penalty and dual will
        /// be ignored.</param>
        /// <param name="fitIntercept">Whether to calculate the intercept for this model. If set
        /// to false, no intercept will be used in calculations
        /// (e.g. data is expected to be already centered).</param>
        /// <param name="interceptScaling">
        ///         when self.fit_intercept is True, instance vector x becomes
        /// [x, self.intercept_scaling],
        /// i.e. a "synthetic" feature with constant value equals to
        /// intercept_scaling is appended to the instance vector.
        /// The intercept becomes intercept_scaling * synthetic feature weight
        /// Note! the synthetic feature weight is subject to l1/l2 regularization
        /// as all other features.
        /// To lessen the effect of regularization on synthetic feature weight
        /// (and therefore on the intercept) intercept_scaling has to be increased
        /// </param>
        /// <param name="classWeight">Set the parameter C of class i to class_weight[i]*C for
        /// SVC. If not given, all classes are supposed to have
        /// weight one. The 'auto' mode uses the values of y to
        /// automatically adjust weights inversely proportional to
        /// class frequencies.</param>
        /// <param name="verbose">Enable verbose output. Note that this setting takes advantage of a
        /// per-process runtime setting in liblinear that, if enabled, may not work
        /// properly in a multithreaded context.</param>
        /// <param name="random">The seed of the pseudo random number generator to use when
        /// shuffling the data.</param>
        public LinearSvc(
            double c = 1.0,
            SvcLoss loss = SvcLoss.L2,
            Norm penalty = Norm.L2,
            bool dual = true,
            double tol = 1E-4,
            Multiclass multiclass = Multiclass.Ovr,
            bool fitIntercept = true,
            double interceptScaling = 1,
            ClassWeight<TLabel> classWeight = null,
            int verbose = 0,
            Random random = null) :base (fitIntercept, classWeight)
        {
            libLinearBase = new LibLinearBase<TLabel>(this, penalty, (Loss)loss, dual, tol, c, multiclass, interceptScaling, classWeight, verbose, random);
        }

        /// <summary>
        /// Reduce X to its most important features.
        /// </summary>
        /// <param name="x">matrix of shape [n_samples, n_features]
        ///    The input samples.</param>
        /// <param name="threshold">The threshold value to use for feature selection. Features whose
        ///    importance is greater or equal are kept while the others are
        ///    discarded. If "median" (resp. "mean"), then the threshold value is
        ///    the median (resp. the mean) of the feature importances. A scaling
        ///    factor (e.g., "1.25*mean") may also be used. If None and if
        ///    available, the object attribute ``threshold`` is used. Otherwise,
        ///     "mean" is used by default.</param>
        /// <returns>
        /// [n_samples, n_selected_features]
        /// The input samples with only the selected features.
        /// </returns>
        public Matrix<double> Transform(Matrix<double> x, double? threshold = null)
        {
                    // Retrieve importance vector
        if hasattr(self, "feature_importances_"):
            importances = self.feature_importances_
            if importances is None:
                raise ValueError("Importance weights not computed. Please set"
                                 " the compute_importances parameter before "
                                 "fit.")

        elif hasattr(self, "coef_"):
            if self.coef_.ndim == 1:
                importances = np.abs(self.coef_)

            else:
                importances = np.sum(np.abs(self.coef_), axis=0)

        else:
            raise ValueError("Missing `feature_importances_` or `coef_`"
                             " attribute, did you forget to set the "
                             "estimator's parameter to compute it?")
        if len(importances) != X.shape[1]:
            raise ValueError("X has different number of features than"
                             " during model fitting.")

        // Retrieve threshold
        if threshold is None:
            if hasattr(self, "penalty") and self.penalty == "l1":
                # the natural default threshold is 0 when l1 penalty was used
                threshold = getattr(self, "threshold", 1e-5)
            else:
                threshold = getattr(self, "threshold", "mean")

        if isinstance(threshold, six.string_types):
            if "*" in threshold:
                scale, reference = threshold.split("*")
                scale = float(scale.strip())
                reference = reference.strip()

                if reference == "median":
                    reference = np.median(importances)
                elif reference == "mean":
                    reference = np.mean(importances)
                else:
                    raise ValueError("Unknown reference: " + reference)

                threshold = scale * reference

            elif threshold == "median":
                threshold = np.median(importances)

            elif threshold == "mean":
                threshold = np.mean(importances)

        else:
            threshold = float(threshold)

        // Selection
        try:
            mask = importances >= threshold
        except TypeError:
            // Fails in Python 3.x when threshold is str;
            // result is array of True
            raise ValueError("Invalid threshold: all features are discarded.")

        if np.any(mask):
            mask = safe_mask(X, mask)
            return X[:, mask]
        else
            raise ValueError("Invalid threshold: all features are discarded.")

        }
    }
}
