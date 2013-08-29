// -----------------------------------------------------------------------
// <copyright file="SolverType.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class SolverType {

        private static readonly Dictionary<int, SolverType> solverTypes = new Dictionary<int, SolverType>
                                                              {
                                                                  {L2R_LR, new SolverType(L2R_LR, "L2R_LR", true, false)},
                                                                  {L2R_L2LOSS_SVC_DUAL, new SolverType(L2R_L2LOSS_SVC_DUAL,"L2R_L2LOSS_SVC_DUAL", false, false)},
                                                                  {L2R_L2LOSS_SVC, new SolverType(L2R_L2LOSS_SVC, "L2R_L2LOSS_SVC", false, false)},
                                                                  {L2R_L1LOSS_SVC_DUAL, new SolverType(L2R_L1LOSS_SVC_DUAL, "L2R_L1LOSS_SVC_DUAL", false, false)},
                                                                  {MCSVM_CS, new SolverType(MCSVM_CS, "MCSVM_CS", false, false)},
                                                                  {L1R_L2LOSS_SVC, new SolverType(L1R_L2LOSS_SVC, "L1R_L2LOSS_SVC", false, false)},
                                                                  {L1R_LR, new SolverType(L1R_LR, "L1R_LR", true, false)},
                                                                  {L2R_LR_DUAL, new SolverType(L2R_LR_DUAL, "L2R_LR_DUAL", true, false)},
                                                                  {L2R_L2LOSS_SVR, new SolverType(L2R_L2LOSS_SVR, "L2R_L2LOSS_SVR", false, true)},
                                                                  {L2R_L2LOSS_SVR_DUAL, new SolverType(L2R_L2LOSS_SVR_DUAL, "L2R_L2LOSS_SVR_DUAL", false, true)},
                                                                  {L2R_L1LOSS_SVR_DUAL, new SolverType(L2R_L1LOSS_SVR_DUAL, "L2R_L1LOSS_SVR_DUAL", false, true)}
                                                              };
    /**
     * L2-regularized logistic regression (primal)
     *
     * (fka L2_LR)
     */
    public const int L2R_LR  = 0;


    /**
     * L2-regularized L2-loss support vector classification (dual)
     *
     * (fka L2LOSS_SVM_DUAL)
     */
    public const int L2R_L2LOSS_SVC_DUAL = 1;


    /**
     * L2-regularized L2-loss support vector classification (primal)
     *
     * (fka L2LOSS_SVM)
     */
    public const int L2R_L2LOSS_SVC = 2;


    /**
     * L2-regularized L1-loss support vector classification (dual)
     *
     * (fka L1LOSS_SVM_DUAL)
     */
     public const int L2R_L1LOSS_SVC_DUAL = 3;


    /**
     * multi-class support vector classification by Crammer and Singer
     */
    public const int MCSVM_CS = 4;


    /**
     * L1-regularized L2-loss support vector classification
     *
     * @since 1.5
     */
    public const int L1R_L2LOSS_SVC = 5;


    /**
     * L1-regularized logistic regression
     *
     * @since 1.5
     */
    public const int L1R_LR = 6;


    /**
     * L2-regularized logistic regression (dual)
     *
     * @since 1.7
     */
    public const int L2R_LR_DUAL = 7;


    /**
     * L2-regularized L2-loss support vector regression (dual)
     *
     * @since 1.91
     */
    public const int L2R_L2LOSS_SVR = 11;


    /**
     * L2-regularized L1-loss support vector regression (dual)
     *
     * @since 1.91
     */
    public const int L2R_L2LOSS_SVR_DUAL = 12;


    /**
     * L2-regularized L2-loss support vector regression (primal)
     *
     * @since 1.91
     */
    public const int L2R_L1LOSS_SVR_DUAL = 13;

    private readonly bool logisticRegressionSolver;
    private readonly bool supportVectorRegression;
    private readonly int     id;


    private SolverType( int id, string name, bool logisticRegressionSolver, bool supportVectorRegression ) {
        this.id = id;
        this.Name = name;
        this.logisticRegressionSolver = logisticRegressionSolver;
        this.supportVectorRegression = supportVectorRegression;
    }

        public string Name
        {
            get; private set;
        }

        public int getId() {
        return id;
    }

    public static SolverType getById(int id)
    {
        SolverType solverType;
        if (!solverTypes.TryGetValue(id, out solverType))
        {
            throw new InvalidOperationException("found no solvertype for id " + id);
        }

        return solverType;
    }

    public static SolverType fromName(string name)
    {
        return solverTypes.Values.FirstOrDefault(s => s.Name.Equals(name, StringComparison.InvariantCultureIgnoreCase));
    }

    public static SolverType[] values()
    {
        return solverTypes.Values.ToArray();
    }


    /**
     * @since 1.9
     */
    public bool isLogisticRegressionSolver() {
        return logisticRegressionSolver;
    }


    /**
     * @since 1.91
     */
    public bool isSupportVectorRegression() {
        return supportVectorRegression;
    }
}

}
