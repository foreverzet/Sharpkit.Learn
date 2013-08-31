// -----------------------------------------------------------------------
// <copyright file="Solver_MCSVM_CS.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

using System.Diagnostics;

namespace Liblinear
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
   /**
 * A coordinate descent algorithm for
 * multi-class support vector machines by Crammer and Singer
 *
 * <pre>
 * min_{\alpha} 0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
 * s.t. \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
 *
 * where e^m_i = 0 if y_i = m,
 * e^m_i = 1 if y_i != m,
 * C^m_i = C if m = y_i,
 * C^m_i = 0 if m != y_i,
 * and w_m(\alpha) = \sum_i \alpha^m_i x_i
 *
 * Given:
 * x, y, C
 * eps is the stopping tolerance
 *
 * solution will be put in w
 *
 * See Appendix of LIBLINEAR paper, Fan et al. (2008)
 * </pre>
 */
class SolverMCSVM_CS {


    private readonly double[] B;
    private readonly double[] C;
    private readonly double   eps;
    private readonly double[] G;
    private readonly int      max_iter;
    private readonly int      w_size, l;
    private readonly int      nr_class;
    private readonly Problem  prob;


    public SolverMCSVM_CS( Problem prob, int nr_class, double[] C ) : this(prob, nr_class, C, 0.1) {
    }


    public SolverMCSVM_CS( Problem prob, int nr_class, double[] C, double eps ) : this(prob, nr_class, C, eps, 100000) {
        
    }




    public SolverMCSVM_CS( Problem prob, int nr_class, double[] weighted_C, double eps, int max_iter ) {
        this.w_size = prob.n;
        this.l = prob.l;
        this.nr_class = nr_class;
        this.eps = eps;
        this.max_iter = max_iter;
        this.prob = prob;
        this.C = weighted_C;
        this.B = new double[nr_class];
        this.G = new double[nr_class];
    }


    private int GETI(int i) {
        return (int)prob.y[i];
    }


    private bool be_shrunk(int i, int m, int yi, double alpha_i, double minG) {
        double bound = 0;
        if (m == yi) bound = C[GETI(i)];
        if (alpha_i == bound && G[m] < minG) return true;
        return false;
    }


    public void solve(double[] w) {
        int i, m, s;
        int iter = 0;
        double[] alpha = new double[l * nr_class];
        double[] alpha_new = new double[nr_class];
        int[] index = new int[l];
        double[] QD = new double[l];
        int[] d_ind = new int[nr_class];
        double[] d_val = new double[nr_class];
        int[] alpha_index = new int[nr_class * l];
        int[] y_index = new int[l];
        int active_size = l;
        int[] active_size_i = new int[l];
        double eps_shrink = Math.Max(10.0 * eps, 1.0); // stopping tolerance for shrinking
        bool start_from_all = true;


        // Initial alpha can be set here. Note that
        // sum_m alpha[i*nr_class+m] = 0, for all i=1,...,l-1
        // alpha[i*nr_class+m] <= C[GETI(i)] if prob->y[i] == m
        // alpha[i*nr_class+m] <= 0 if prob->y[i] != m
        // If initial alpha isn't zero, uncomment the for loop below to initialize w
        for (i = 0; i < l * nr_class; i++)
            alpha[i] = 0;


        for (i = 0; i < w_size * nr_class; i++)
            w[i] = 0;
        for (i = 0; i < l; i++) {
            for (m = 0; m < nr_class; m++)
                alpha_index[i * nr_class + m] = m;
            QD[i] = 0;
            foreach (Feature xi in prob.x[i]) {
                double val = xi.Value;
                QD[i] += val * val;


                // Uncomment the for loop if initial alpha isn't zero
                // for(m=0; m<nr_class; m++)
                //  w[(xi->index-1)*nr_class+m] += alpha[i*nr_class+m]*val;
            }
            active_size_i[i] = nr_class;
            y_index[i] = (int)prob.y[i];
            index[i] = i;
        }


        ArrayPointer<double> alpha_i = new ArrayPointer<double>(alpha, 0);
        ArrayPointer<int> alpha_index_i = new ArrayPointer<int>(alpha_index, 0);


        while (iter < max_iter) {
            double stopping = Double.NegativeInfinity;


            for (i = 0; i < active_size; i++) {
                // int j = i+rand()%(active_size-i);
                int j = i + Linear.random.Next(active_size - i);
                Linear.swap(index, i, j);
            }
            for (s = 0; s < active_size; s++) {


                i = index[s];
                double Ai = QD[i];
                // double *alpha_i = &alpha[i*nr_class];
                alpha_i.setOffset(i * nr_class);


                // int *alpha_index_i = &alpha_index[i*nr_class];
                alpha_index_i.setOffset(i * nr_class);


                if (Ai > 0) {
                    for (m = 0; m < active_size_i[i]; m++)
                        G[m] = 1;
                    if (y_index[i] < active_size_i[i]) G[y_index[i]] = 0;


                    foreach (Feature xi in prob.x[i]) {
                        // double *w_i = &w[(xi.index-1)*nr_class];
                        int w_offset = (xi.Index - 1) * nr_class;
                        for (m = 0; m < active_size_i[i]; m++)
                            // G[m] += w_i[alpha_index_i[m]]*(xi.value);
                            G[m] += w[w_offset + alpha_index_i[m]] * (xi.Value);


                    }


                    double minG = Double.PositiveInfinity;
                    double maxG = Double.NegativeInfinity;
                    for (m = 0; m < active_size_i[i]; m++) {
                        if (alpha_i[alpha_index_i[m]] < 0 && G[m] < minG) minG = G[m];
                        if (G[m] > maxG) maxG = G[m];
                    }
                    if (y_index[i] < active_size_i[i]) {
                        if (alpha_i[(int)prob.y[i]] < C[GETI(i)] && G[y_index[i]] < minG) {
                            minG = G[y_index[i]];
                        }
                    }


                    for (m = 0; m < active_size_i[i]; m++) {
                        if (be_shrunk(i, m, y_index[i], alpha_i[alpha_index_i[m]], minG)) {
                            active_size_i[i]--;
                            while (active_size_i[i] > m) {
                                if (!be_shrunk(i, active_size_i[i], y_index[i], alpha_i[alpha_index_i[active_size_i[i]]], minG)) {
                                    Linear.swap(alpha_index_i, m, active_size_i[i]);
                                    Linear.swap(G, m, active_size_i[i]);
                                    if (y_index[i] == active_size_i[i])
                                        y_index[i] = m;
                                    else if (y_index[i] == m) y_index[i] = active_size_i[i];
                                    break;
                                }
                                active_size_i[i]--;
                            }
                        }
                    }


                    if (active_size_i[i] <= 1) {
                        active_size--;
                        Linear.swap(index, s, active_size);
                        s--;
                        continue;
                    }


                    if (maxG - minG <= 1e-12)
                        continue;
                    else
                        stopping = Math.Max(maxG - minG, stopping);


                    for (m = 0; m < active_size_i[i]; m++)
                        B[m] = G[m] - Ai * alpha_i[alpha_index_i[m]];


                    solve_sub_problem(Ai, y_index[i], C[GETI(i)], active_size_i[i], alpha_new);
                    int nz_d = 0;
                    for (m = 0; m < active_size_i[i]; m++) {
                        double d = alpha_new[m] - alpha_i[alpha_index_i[m]];
                        alpha_i[alpha_index_i[m]] = alpha_new[m];
                        if (Math.Abs(d) >= 1e-12) {
                            d_ind[nz_d] = alpha_index_i[m];
                            d_val[nz_d] = d;
                            nz_d++;
                        }
                    }


                    foreach (Feature xi in prob.x[i]) {
                        // double *w_i = &w[(xi->index-1)*nr_class];
                        int w_offset = (xi.Index - 1) * nr_class;
                        for (m = 0; m < nz_d; m++) {
                            w[w_offset + d_ind[m]] += d_val[m] * xi.Value;
                        }
                    }
                }
            }


            iter++;


            if (iter % 10 == 0) {
                Linear.info(".");
            }


            if (stopping < eps_shrink) {
                if (stopping < eps && start_from_all == true)
                    break;
                else {
                    active_size = l;
                    for (i = 0; i < l; i++)
                        active_size_i[i] = nr_class;
                    Linear.info("*");
                    eps_shrink = Math.Max(eps_shrink / 2, eps);
                    start_from_all = true;
                }
            } else
                start_from_all = false;
        }


        Linear.info("\noptimization finished, #iter = {0}", iter);
        if (iter >= max_iter) Linear.info("\nWARNING: reaching max number of iterations");


        // calculate objective value
        double v = 0;
        int nSV = 0;
        for (i = 0; i < w_size * nr_class; i++)
            v += w[i] * w[i];
        v = 0.5 * v;
        for (i = 0; i < l * nr_class; i++) {
            v += alpha[i];
            if (Math.Abs(alpha[i]) > 0) nSV++;
        }
        for (i = 0; i < l; i++)
            v -= alpha[i * nr_class + (int)prob.y[i]];
        Linear.info("Objective value = {0}", v);
        Linear.info("nSV = {0}", nSV);


    }


    private void solve_sub_problem(double A_i, int yi, double C_yi, int active_i, double[] alpha_new) {


        int r;
        Debug.Assert(active_i <= B.Length); // no padding
        double[] D = Linear.copyOf(B, active_i);
        // clone(D, B, active_i);


        if (yi < active_i) D[yi] += A_i * C_yi;


        // qsort(D, active_i, sizeof(double), compare_double);
        ArraySorter.reversedMergesort(D);


        double beta = D[0] - A_i * C_yi;
        for (r = 1; r < active_i && beta < r * D[r]; r++)
            beta += D[r];
        beta /= r;


        for (r = 0; r < active_i; r++) {
            if (r == yi)
                alpha_new[r] = Math.Min(C_yi, (beta - B[r]) / A_i);
            else
                alpha_new[r] = Math.Min(0.0, (beta - B[r]) / A_i);
        }
    }
}

}
