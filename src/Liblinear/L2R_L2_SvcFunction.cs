// -----------------------------------------------------------------------
// <copyright file="L2R_L2_SvcFunction.cs" company="">
// TODO: Update copyright text.
// </copyright>
// -----------------------------------------------------------------------

namespace Liblinear
{
    using System;

    /// <summary>
    /// TODO: Update summary.
    /// </summary>
    public class L2R_L2_SvcFunction : IFunction
    {
        protected readonly Problem prob;
        protected readonly double[] C;
        protected readonly int[] I;
        protected readonly double[] z;

        protected int sizeI;

        public L2R_L2_SvcFunction(Problem prob, double[] C)
        {
            int l = prob.l;

            this.prob = prob;

            z = new double[l];
            I = new int[l];
            this.C = C;
        }


        public virtual double fun(double[] w)
        {
            int i;
            double f = 0;
            double[] y = prob.y;
            int l = prob.l;
            int w_size = get_nr_variable();

            Xv(w, z);

            for (i = 0; i < w_size; i++)
                f += w[i] * w[i];
            f /= 2.0;
            for (i = 0; i < l; i++)
            {
                z[i] = y[i] * z[i];
                double d = 1 - z[i];
                if (d > 0) f += C[i] * d * d;
            }

            return (f);
        }

        public int get_nr_variable()
        {
            return prob.n;
        }

        public virtual void grad(double[] w, double[] g)
        {
            double[] y = prob.y;
            int l = prob.l;
            int w_size = get_nr_variable();


            sizeI = 0;
            for (int i = 0; i < l; i++)
            {
                if (z[i] < 1)
                {
                    z[sizeI] = C[i] * y[i] * (z[i] - 1);
                    I[sizeI] = i;
                    sizeI++;
                }
            }
            subXTv(z, g);


            for (int i = 0; i < w_size; i++)
                g[i] = w[i] + 2 * g[i];
        }

        public void Hv(double[] s, double[] Hs)
        {
            int i;
            int w_size = get_nr_variable();
            double[] wa = new double[sizeI];


            subXv(s, wa);
            for (i = 0; i < sizeI; i++)
                wa[i] = C[I[i]] * wa[i];


            subXTv(wa, Hs);
            for (i = 0; i < w_size; i++)
                Hs[i] = s[i] + 2 * Hs[i];
        }

        protected void subXTv(double[] v, double[] XTv)
        {
            int i;
            int w_size = get_nr_variable();


            for (i = 0; i < w_size; i++)
                XTv[i] = 0;


            for (i = 0; i < sizeI; i++)
            {
                foreach (Feature s in prob.x[I[i]])
                {
                    XTv[s.Index - 1] += v[i] * s.Value;
                }
            }
        }

        private void subXv(double[] v, double[] Xv)
        {
            for (int i = 0; i < sizeI; i++)
            {
                Xv[i] = 0;
                foreach (Feature s in prob.x[I[i]])
                {
                    Xv[i] += v[s.Index - 1] * s.Value;
                }
            }
        }

        protected void Xv(double[] v, double[] Xv)
        {
            for (int i = 0; i < prob.l; i++)
            {
                Xv[i] = 0;
                foreach (Feature s in prob.x[i])
                {
                    Xv[i] += v[s.Index - 1] * s.Value;
                }
            }
        }
    }
}
