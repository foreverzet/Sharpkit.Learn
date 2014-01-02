// -----------------------------------------------------------------------
// <copyright file="Tree.cs" company="Sharpkit.Learn">
//  Authors: Gilles Louppe <g.louppe@gmail.com>
//           Peter Prettenhofer <peter.prettenhofer@gmail.com>
//           Brian Holt <bdholt1@gmail.com>
//           Noel Dawe <noel@dawe.me>
//           Satrajit Gosh <satrajit.ghosh@gmail.com>
//           Lars Buitinck <L.J.Buitinck@uva.nl>
//           Sergey Zyuzin
//  Licence: BSD 3 clause
// </copyright>
// -----------------------------------------------------------------------

namespace Sharpkit.Learn.Tree
{
    using System;
    using System.Linq;
    using MathNet.Numerics.LinearAlgebra.Double;

    /// <summary>
    /// Struct-of-arrays representation of a binary decision tree.
    ///
    /// The binary tree is represented as a number of parallel arrays. The i-th
    /// element of each array holds information about the node `i`. Node 0 is the
    /// tree's root. You can find a detailed description of all arrays in
    /// `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    /// nodes, resp. In this case the values of nodes of the other type are
    /// arbitrary!
    /// </summary>
    /// <remarks>
    /// https://github.com/scikit-learn/scikit-learn/tree/30eb78de8d1e7b25fb1a4b0d8c63afdcc972ee84/sklearn/tree/_tree.pyx
    /// </remarks>
    internal class Tree
    {
        private const uint _TREE_LEAF = 0xFFFFFFFF;
        private const uint _TREE_UNDEFINED = 0xFFFFFFFE;

        /// <summary>
        /// The current capacity (i.e., size) of the arrays.
        /// </summary>
        public int capacity { get; private set; }

        /// <summary>
        ///         The number of nodes (internal nodes + leaves) in the tree.
        /// </summary>
        public uint node_count { get; private set; }
        public uint[] n_classes { get; private set; }

        /// <summary>
        /// children_left[i] holds the node id of the left child of node i.
        /// For leaves, children_left[i] == TREE_LEAF. Otherwise,
        /// children_left[i] > i. This child handles the case where
        /// X[:, feature[i]] <= threshold[i].
        /// </summary>
        public uint[] children_left { get; private set; }

        /// <summary>
        ///         children_right[i] holds the node id of the right child of node i.
        /// For leaves, children_right[i] == TREE_LEAF. Otherwise,
        /// children_right[i] > i. This child handles the case where
        /// X[:, feature[i]] > threshold[i].
        /// </summary>
        public uint[] children_right { get; private set; }

        /// <summary>
        /// feature[i] holds the feature to split on, for the internal node i.
        /// </summary>
        public uint[] feature { get; private set; }

        /// <summary>
        /// impurity[i] holds the impurity (i.e., the value of the splitting
        /// criterion) at node i.
        /// </summary>
        public double[] impurity { get; private set; }

        /// <summary>
        /// threshold[i] holds the threshold for the internal node i.
        /// </summary>
        public double[] threshold { get; private set; }

        /// <summary>
        /// Contains the constant prediction value of each node.
        /// </summary>
        public double[] value { get; private set; }

        /// <summary>
        /// n_samples[i] holds the number of training samples reaching node i.
        /// </summary>
        public uint[] n_node_samples { get; private set; }

        private int n_features;
        private int n_outputs;
        private uint max_n_classes;
        private uint value_stride;
        private SplitterBase splitter;
        private uint max_depth;
        private uint min_samples_split;
        private uint min_samples_leaf;

    public Tree(int n_features, uint[] n_classes,
                        int n_outputs, SplitterBase splitter, uint max_depth,
                        uint min_samples_split, uint min_samples_leaf)
    {
        // Input/Output layout
        this.n_features = n_features;
        this.n_outputs = n_outputs;
        this.n_classes = new uint[n_outputs];


        this.max_n_classes = n_classes.Max();
        this.value_stride = (uint)this.n_outputs*this.max_n_classes;


        for (uint k =0; k < n_outputs; k++)
        {
            this.n_classes[k] = n_classes[k];
        }
    
        // Parameters
        this.splitter = splitter;
        this.max_depth = max_depth;
        this.min_samples_split = min_samples_split;
        this.min_samples_leaf = min_samples_leaf;

        // Inner structures
        this.node_count = 0;
        this.capacity = 0;
        this.children_left = null;
        this.children_right = null;
        this.feature = null;
        this.threshold = null;
        this.value = null;
        this.impurity = null;
        this.n_node_samples = null;
    }

    private T[] Resize<T>(T[] arr, int newLength)
    {
        var tmpNew = new T[newLength];
        if (arr != null)
        {
            Array.Copy(arr, tmpNew, Math.Min(arr.Length, tmpNew.Length));
        }

        return tmpNew;
    }

    /// <summary>
    /// Resize all inner arrays to `capacity`, if `capacity` < 0, then
    ///       double the size of the inner arrays.
    /// </summary>
    /// <param name="capacity"></param>
    public void _resize(int capacity=-1)
    {
        if (capacity == this.capacity)
        {
            return;
        }

        if (capacity < 0)
        {
            if (this.capacity <= 0)
            {
                capacity = 3; // default initial value
            }
            else
            {
                capacity = 2*this.capacity;
            }
        }

        this.capacity = capacity;

        this.children_left = Resize(children_left, capacity);
        this.children_right = Resize(this.children_right, capacity);
        this.feature = Resize(this.feature, capacity);
        this.threshold = Resize(this.threshold, capacity);
        this.value = Resize(this.value, capacity * (int)this.value_stride);
        this.impurity = Resize(this.impurity, capacity);
        this.n_node_samples = Resize(this.n_node_samples, capacity);

        // if capacity smaller than node_count, adjust the counter
        if (capacity < this.node_count)
        {
            this.node_count = (uint)capacity;
        }
    }

    /// <summary>
    /// Add a node to the tree. The new node registers itself as
    ///       the child of its parent. 
    /// </summary>
    /// <param name="parent"></param>
    /// <param name="is_left"></param>
    /// <param name="is_leaf"></param>
    /// <param name="feature"></param>
    /// <param name="threshold"></param>
    /// <param name="impurity"></param>
    /// <param name="n_node_samples"></param>
    /// <returns></returns>
        private uint _add_node(uint parent,
                                bool is_left,
                                bool is_leaf,
                                uint feature,
                                double threshold,
                                double impurity,
                                uint n_node_samples)
    {
        uint node_id = this.node_count;


        if (node_id >= this.capacity)
        {
            this._resize();
        }

        this.impurity[node_id] = impurity;
        this.n_node_samples[node_id] = n_node_samples;

        if (parent != _TREE_UNDEFINED)
        {
            if (is_left)
            {
                this.children_left[parent] = node_id;
            }
            else
            {
                this.children_right[parent] = node_id;
            }
        }


        if (is_leaf)
        {
            this.children_left[node_id] = _TREE_LEAF;
            this.children_right[node_id] = _TREE_LEAF;
            this.feature[node_id] = _TREE_UNDEFINED;
            this.threshold[node_id] = _TREE_UNDEFINED;
        }
        else
        {
            // children_left and children_right will be set later
            this.feature[node_id] = feature;
            this.threshold[node_id] = threshold;
        }

        this.node_count += 1;


        return node_id;
    }

    
    /// <summary>
    /// Build a decision tree from the training set (X, y).
    /// </summary>
    public void build(MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> X,
                      MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> y,
                      MathNet.Numerics.LinearAlgebra.Generic.Vector<double> sample_weight = null)
    {
        // Prepare data before recursive partitioning

        // Initial capacity
        int init_capacity;


        if (this.max_depth <= 10)
        {
            init_capacity = (int)Math.Pow(2, (this.max_depth + 1)) - 1;
        }
        else
        {
            init_capacity = 2047;
        }

        this._resize(init_capacity);


        // Recursive partition (without actual recursion)
        SplitterBase splitter = this.splitter;
        splitter.init(X, y, sample_weight == null ? null : sample_weight.ToArray());


        uint stack_n_values = 5;
        uint stack_capacity = 50;
        uint[] stack = new uint[stack_capacity];


        stack[0] = 0;                    // start
        stack[1] = splitter.n_samples;   // end
        stack[2] = 0;                    // depth
        stack[3] = _TREE_UNDEFINED;      // parent
        stack[4] = 0;                    // is_left


        uint start;
        uint end;
        uint depth;
        uint parent;
        bool is_left;


        uint n_node_samples;
        uint pos = 0;
        uint feature = 0;
        double threshold = 0;
        double impurity = 0;
        bool is_leaf = false;

        uint node_id;

        while (stack_n_values > 0)
        {
            stack_n_values -= 5;


            start = stack[stack_n_values];
            end = stack[stack_n_values + 1];
            depth = stack[stack_n_values + 2];
            parent = stack[stack_n_values + 3];
            is_left = stack[stack_n_values + 4] != 0;


            n_node_samples = end - start;
            is_leaf = ((depth >= this.max_depth) ||
                       (n_node_samples < this.min_samples_split) ||
                       (n_node_samples < 2*this.min_samples_leaf));


            splitter.node_reset(start, end, ref impurity);
            is_leaf = is_leaf || (impurity < 1e-7);


            if (!is_leaf)
            {
                splitter.node_split(ref pos, ref feature, ref threshold);
                is_leaf = is_leaf || (pos >= end);
            }


            node_id = this._add_node(parent, is_left, is_leaf, feature,
                                     threshold, impurity, n_node_samples);


            if (is_leaf)
            {
                // Don't store value for internal nodes
                splitter.node_value(this.value, node_id*this.value_stride);
            }
            else
            {
                if (stack_n_values + 10 > stack_capacity)
                {
                    stack_capacity *= 2;
                    var newStack = new uint[stack_capacity];
                    Array.Copy(stack, newStack, stack.Length);
                    stack = newStack;
                }


                // Stack right child
                stack[stack_n_values] = pos;
                stack[stack_n_values + 1] = end;
                stack[stack_n_values + 2] = depth + 1;
                stack[stack_n_values + 3] = node_id;
                stack[stack_n_values + 4] = 0;
                stack_n_values += 5;


                // Stack left child
                stack[stack_n_values] = start;
                stack[stack_n_values + 1] = pos;
                stack[stack_n_values + 2] = depth + 1;
                stack[stack_n_values + 3] = node_id;
                stack[stack_n_values + 4] = 1;
                stack_n_values += 5;
            }

            this._resize((int)this.node_count);
        }

        this.splitter = null; // Release memory
    }
///        """Predict target for X."""
    public MathNet.Numerics.LinearAlgebra.Generic.Matrix<double>[] predict(MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> X)
    {
        uint n_samples = (uint)X.RowCount;

        if (n_outputs == 1)
        {
            MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> @out =
                DenseMatrix.Create((int)n_samples, (int)max_n_classes, (i, j) => 0.0);
            

            for (int i = 0; i < n_samples; i++)
            {
                uint node_id = 0;

                // While node_id not a leaf
                while (children_left[node_id] != _TREE_LEAF)
                {
                    // ... and children_right[node_id] != _TREE_LEAF:
                    if (X[i, (int)feature[node_id]] <= threshold[node_id])
                    {
                        node_id = children_left[node_id];
                    }
                    else
                    {
                        node_id = children_right[node_id];
                    }
                }

                uint offset = node_id*value_stride;


                for (int c = 0; c < n_classes[0]; c++)
                {
                    @out[i, c] = value[offset + c];
                }
            }

            return new[]{@out};

        }
        else // n_outputs > 1
        {
            //out_multi = np.zeros((n_samples,
            //                      n_outputs,
            //                      max_n_classes), dtype=np.float64)
            //Note: I've changed order
            var out_multi = new MathNet.Numerics.LinearAlgebra.Generic.Matrix<double>[n_outputs];
            for (int i=0; i<out_multi.Length; i++)
            {
                out_multi[i] = DenseMatrix.Create((int)n_samples, (int)max_n_classes, (c, r) => 0.0);
            }

            for (int i=0; i < n_samples; i++)
            {
                uint node_id = 0;

                // While node_id not a leaf
                while (children_left[node_id] != _TREE_LEAF)
                {
                    // ... and children_right[node_id] != _TREE_LEAF:
                    if (X[i, (int)feature[node_id]] <= threshold[node_id])
                    {
                        node_id = children_left[node_id];
                    }
                    else
                    {
                        node_id = children_right[node_id];
                    }
                }

                uint offset = node_id*value_stride;


                for (int k=0; k < n_outputs; k++)
                {
                    for (int c = 0; c < n_classes[k]; c++)
                    {
                        //out_multi[i, k, c] = value[offset + c];
                        //Note: I've changed order
                        out_multi[k][i, c] = value[offset + c];
                    }
                    offset += max_n_classes;
                }
            }

            return out_multi;

        }
    }

    ///"""Finds the terminal region (=leaf node) for each sample in X."""
    public MathNet.Numerics.LinearAlgebra.Generic.Vector<double> apply(MathNet.Numerics.LinearAlgebra.Generic.Matrix<double> X)
    {
        //cdef double* threshold = this.threshold

        uint n_samples = (uint)X.RowCount;


        MathNet.Numerics.LinearAlgebra.Generic.Vector<double> @out = DenseVector.Create((int)n_samples, i => 0.0);
        
        for ( int i = 0; i < n_samples; i++)
        {
            uint node_id = 0;


            // While node_id not a leaf
            while (children_left[node_id] != _TREE_LEAF)
            {
                // ... and children_right[node_id] != _TREE_LEAF:
                if (X[i, (int)feature[node_id]] <= threshold[node_id])
                {
                    node_id = children_left[node_id];
                }
                else
                {
                    node_id = children_right[node_id];
                }
            }

            @out[i] = node_id;
        }

        return @out;
    }

     /// <summary>
     /// Computes the importance of each feature (aka variable).
     /// </summary>
     /// <param name="normalize"></param>
     public  MathNet.Numerics.LinearAlgebra.Generic.Vector<double> compute_feature_importances(bool normalize=true)
     {
         MathNet.Numerics.LinearAlgebra.Generic.Vector<double> importances = DenseVector.Create(this.n_features, i => 0.0);


        for (uint node = 0; node < node_count; node++)
        {
            if (children_left[node] != _TREE_LEAF)
            {
                // ... and children_right[node] != _TREE_LEAF:
                uint n_left = n_node_samples[children_left[node]];
                uint n_right = n_node_samples[children_right[node]];


                importances[(int)feature[node]] +=
                    n_node_samples[node]*impurity[node]
                    - n_left*impurity[children_left[node]]
                    - n_right*impurity[children_right[node]];
            }
        }

         importances = importances.Divide(this.n_node_samples[0]);
         double normalizer = 0;


        if (normalize)
        {
            normalizer = importances.Sum();
        }


         if (normalizer > 0.0)
         {
             //   Avoid dividing by zero (e.g., when root is pure)
             importances /= normalizer;
         }


         return importances;
     }
    }
}
