import random
import unittest

from matvec_multiply import dot_product, matvec_multiply


# generate tests for matrix-vector-product function in matvec_multiply.py
class TestMatVecMultiply(unittest.TestCase):
    """Tests for matvec_multiply, covering shapes, special matrices, and errors."""

    def test_small_known_product(self):
        # Hand-computed example: rows [1,2,3] and [4,5,6] against [1,0,-1].
        A = [[1, 2, 3], [4, 5, 6]]
        x = [1, 0, -1]
        self.assertEqual(matvec_multiply(A, x), [-2.0, -2.0])

    def test_identity_matrix_returns_input_vector(self):
        # Multiplying by the identity must reproduce x exactly.
        n = 5
        I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        x = [2.5, -1.0, 0.0, 7.25, 3.0]
        self.assertEqual(matvec_multiply(I, x), x)

    def test_zero_matrix_returns_zero_vector(self):
        # A matrix of zeros annihilates any vector.
        A = [[0, 0, 0] for _ in range(4)]
        x = [1.5, -2.5, 9.0]
        self.assertEqual(matvec_multiply(A, x), [0.0, 0.0, 0.0, 0.0])

    def test_zero_vector_returns_zero_vector(self):
        # A zero vector gives a zero result regardless of the matrix.
        A = [[1, 2], [3, 4], [5, 6]]
        x = [0, 0]
        self.assertEqual(matvec_multiply(A, x), [0.0, 0.0, 0.0])

    def test_non_square_matrix_result_length(self):
        # The result has one entry per row, not per column.
        A = [[1, 2, 3, 4], [5, 6, 7, 8]]
        x = [1, 1, 1, 1]
        y = matvec_multiply(A, x)
        self.assertEqual(len(y), 2)
        self.assertEqual(y, [10.0, 26.0])

    def test_single_row_matches_dot_product(self):
        # A one-row matrix reduces to a single dot product.
        row = [3, -1, 4]
        x = [2, 5, -2]
        self.assertEqual(matvec_multiply([row], x), [-7.0])

    def test_empty_matrix_returns_empty_list(self):
        # No rows means no output entries.
        self.assertEqual(matvec_multiply([], [1, 2, 3]), [])

    def test_linearity_in_the_vector(self):
        # A(u + v) equals Au + Av for the same matrix A.
        A = [[1, -2, 3], [0, 4, -1]]
        u = [1.0, 2.0, 3.0]
        v = [-0.5, 1.5, 2.0]
        w = [u[i] + v[i] for i in range(3)]
        left = matvec_multiply(A, w)
        right_a = matvec_multiply(A, u)
        right_b = matvec_multiply(A, v)
        for i in range(len(left)):
            self.assertAlmostEqual(left[i], right_a[i] + right_b[i], places=10)

    def test_matches_independent_computation_on_random_data(self):
        # Compare against sum over zip, which does not use the functions under test.
        random.seed(1)
        rows, cols = 40, 25
        A = [[random.uniform(-5.0, 5.0) for _ in range(cols)] for _ in range(rows)]
        x = [random.uniform(-5.0, 5.0) for _ in range(cols)]
        y = matvec_multiply(A, x)
        expected = [sum(a * b for a, b in zip(row, x)) for row in A]
        for i in range(rows):
            self.assertAlmostEqual(y[i], expected[i], places=10)

    def test_row_length_mismatch_raises(self):
        # A row shorter than x is not a valid matrix for this product.
        A = [[1, 2, 3], [4, 5]]
        with self.assertRaises(ValueError):
            matvec_multiply(A, [1, 1, 1])

    def test_vector_length_mismatch_raises(self):
        # The vector must have one entry per column.
        A = [[1, 2, 3], [4, 5, 6]]
        with self.assertRaises(ValueError):
            matvec_multiply(A, [1, 1])

    def test_input_is_not_modified(self):
        # The function must leave its arguments untouched.
        A = [[1, 2], [3, 4]]
        x = [5, 6]
        matvec_multiply(A, x)
        self.assertEqual(A, [[1, 2], [3, 4]])
        self.assertEqual(x, [5, 6])


# generate tests for dot-product function in matvec_multiply.py
class TestDotProduct(unittest.TestCase):
    """Tests for dot_product, covering values, algebraic properties, and errors."""

    def test_known_value(self):
        # 1*4 + 2*5 + 3*6 = 32.
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32.0)

    def test_with_negative_and_float_values(self):
        # Mixed signs and floats: -1.5*2 + 4*(-0.5) = -5.0.
        self.assertAlmostEqual(dot_product([-1.5, 4.0], [2.0, -0.5]), -5.0, places=10)

    def test_zero_vector_gives_zero(self):
        # Any vector dotted with zeros is zero.
        self.assertEqual(dot_product([1, 2, 3], [0, 0, 0]), 0.0)

    def test_orthogonal_vectors_give_zero(self):
        # Perpendicular axis vectors have no overlap.
        self.assertEqual(dot_product([1, 0], [0, 1]), 0.0)

    def test_single_element_vectors(self):
        # The one-dimensional case is ordinary multiplication.
        self.assertEqual(dot_product([7], [3]), 21.0)

    def test_empty_vectors_give_zero(self):
        # The empty sum is zero.
        self.assertEqual(dot_product([], []), 0.0)

    def test_commutative(self):
        # u . v equals v . u.
        u = [2.0, -3.0, 0.5]
        v = [1.0, 4.0, -2.0]
        self.assertAlmostEqual(dot_product(u, v), dot_product(v, u), places=10)

    def test_self_dot_is_squared_norm(self):
        # u . u equals the sum of squares, which is nonnegative.
        u = [3.0, 4.0]
        self.assertAlmostEqual(dot_product(u, u), 25.0, places=10)

    def test_scaling_one_argument_scales_result(self):
        # Scaling u by c scales the dot product by c.
        u = [1.0, -2.0, 3.0]
        v = [4.0, 5.0, -6.0]
        c = 2.5
        scaled = [c * value for value in u]
        self.assertAlmostEqual(dot_product(scaled, v), c * dot_product(u, v), places=10)

    def test_matches_independent_computation_on_random_data(self):
        # Compare against sum over zip, which does not use the function under test.
        random.seed(2)
        n = 500
        u = [random.uniform(-10.0, 10.0) for _ in range(n)]
        v = [random.uniform(-10.0, 10.0) for _ in range(n)]
        expected = sum(a * b for a, b in zip(u, v))
        self.assertAlmostEqual(dot_product(u, v), expected, places=8)

    def test_length_mismatch_raises(self):
        # Vectors of different lengths have no defined dot product.
        with self.assertRaises(ValueError):
            dot_product([1, 2, 3], [1, 2])

    def test_input_is_not_modified(self):
        # The function must leave its arguments untouched.
        u = [1, 2, 3]
        v = [4, 5, 6]
        dot_product(u, v)
        self.assertEqual(u, [1, 2, 3])
        self.assertEqual(v, [4, 5, 6])


# generate tests for the guards that reject malformed input to both functions
class TestDotProductGuards(unittest.TestCase):
    """Tests for the input guards on dot_product."""

    def test_none_argument_raises_type_error(self):
        # A scalar or None has no length, and must not surface a raw len() error.
        with self.assertRaises(TypeError):
            dot_product(None, [1, 2])
        with self.assertRaises(TypeError):
            dot_product([1, 2], None)

    def test_scalar_argument_raises_type_error(self):
        # An int is not a vector even though the arithmetic would almost work.
        with self.assertRaises(TypeError):
            dot_product(3, [1, 2])

    def test_string_argument_raises_type_error(self):
        # Strings are indexable and sized, so they must be rejected explicitly.
        with self.assertRaises(TypeError):
            dot_product("ab", "cd")
        with self.assertRaises(TypeError):
            dot_product(b"ab", [1, 2])

    def test_set_argument_raises_type_error(self):
        # A set has a length but cannot be indexed positionally.
        with self.assertRaises(TypeError):
            dot_product({1, 2}, [1, 2])

    def test_generator_argument_raises_type_error(self):
        # A generator would be consumed by the length check and read as empty.
        with self.assertRaises(TypeError):
            dot_product((i for i in range(3)), [1, 2, 3])

    def test_non_numeric_element_raises_type_error(self):
        # Nested lists, strings and None inside a vector are all rejected.
        with self.assertRaises(TypeError):
            dot_product([[1, 2]], [3])
        with self.assertRaises(TypeError):
            dot_product([1, "2"], [3, 4])
        with self.assertRaises(TypeError):
            dot_product([1, None], [3, 4])

    def test_complex_element_raises_type_error(self):
        # The real dot product is the wrong inner product for complex vectors.
        with self.assertRaises(TypeError):
            dot_product([1 + 2j], [3])

    def test_nan_element_raises_value_error(self):
        # nan would propagate silently to a nan result.
        with self.assertRaises(ValueError):
            dot_product([1.0, float("nan")], [1.0, 1.0])

    def test_infinite_element_raises_value_error(self):
        # inf would propagate silently, and inf * 0 would produce nan.
        with self.assertRaises(ValueError):
            dot_product([float("inf")], [1.0])
        with self.assertRaises(ValueError):
            dot_product([1.0], [float("-inf")])

    def test_overflow_raises_overflow_error(self):
        # Finite inputs whose products exceed the float range must not return inf.
        with self.assertRaises(OverflowError):
            dot_product([1e308, 1e308], [10.0, 10.0])

    def test_length_mismatch_message_reports_lengths(self):
        # The message should say what the two lengths actually were.
        with self.assertRaises(ValueError) as caught:
            dot_product([1, 2, 3], [1, 2])
        self.assertIn("3", str(caught.exception))
        self.assertIn("2", str(caught.exception))

    def test_error_message_names_the_offending_index(self):
        # A bad element must be locatable by argument name and position.
        with self.assertRaises(TypeError) as caught:
            dot_product([1.0, 2.0, "x"], [1.0, 2.0, 3.0])
        self.assertIn("u[2]", str(caught.exception))

    def test_compensated_summation_keeps_precision(self):
        # Naive left to right accumulation returns 0.0 here; the exact answer is 2.0.
        u = [1e16, 1.0, 1.0, -1e16]
        v = [1.0, 1.0, 1.0, 1.0]
        self.assertEqual(dot_product(u, v), 2.0)

    def test_tuples_are_accepted(self):
        # The guard rejects wrong types without rejecting valid sequence types.
        self.assertEqual(dot_product((1, 2, 3), (4, 5, 6)), 32.0)

    def test_integers_and_booleans_are_accepted(self):
        # bool is a subclass of int and is a legitimate 0 or 1 entry.
        self.assertEqual(dot_product([True, False, True], [2, 3, 4]), 6.0)

    def test_validate_false_skips_checks(self):
        # The opt-out used internally by matvec_multiply must still compute normally.
        self.assertEqual(dot_product([1, 2], [3, 4], validate=False), 11.0)


class TestMatVecMultiplyGuards(unittest.TestCase):
    """Tests for the input guards on matvec_multiply."""

    def test_empty_matrix_still_validates_the_vector(self):
        # The empty-matrix shortcut must not let a malformed x through unchecked.
        with self.assertRaises(TypeError):
            matvec_multiply([], None)
        with self.assertRaises(TypeError):
            matvec_multiply([], "abc")
        with self.assertRaises(ValueError):
            matvec_multiply([], [float("nan")])

    def test_empty_matrix_with_valid_vector_still_returns_empty(self):
        # A valid call must keep working after the added checks.
        self.assertEqual(matvec_multiply([], [1, 2, 3]), [])

    def test_non_sequence_matrix_raises_type_error(self):
        # None, a scalar or a generator is not a matrix.
        with self.assertRaises(TypeError):
            matvec_multiply(None, [1, 2])
        with self.assertRaises(TypeError):
            matvec_multiply(7, [1, 2])
        with self.assertRaises(TypeError):
            matvec_multiply((r for r in [[1, 2]]), [1, 2])

    def test_string_matrix_raises_type_error(self):
        # A string is indexable and sized, so it must be rejected explicitly.
        with self.assertRaises(TypeError):
            matvec_multiply("ab", [1, 2])

    def test_flat_vector_as_matrix_raises_type_error(self):
        # Passing [1,2,3] where [[1,2,3]] was meant is a common mistake.
        with self.assertRaises(TypeError) as caught:
            matvec_multiply([1, 2, 3], [1, 2, 3])
        self.assertIn("A[0]", str(caught.exception))

    def test_non_numeric_entry_reports_row_and_column(self):
        # A single bad cell must be locatable in a large matrix.
        A = [[1, 2], [3, "x"]]
        with self.assertRaises(TypeError) as caught:
            matvec_multiply(A, [1, 1])
        self.assertIn("A[1][1]", str(caught.exception))

    def test_non_finite_entry_raises_value_error(self):
        # nan and inf in the matrix must be refused, not propagated.
        with self.assertRaises(ValueError):
            matvec_multiply([[1.0, float("nan")]], [1.0, 1.0])
        with self.assertRaises(ValueError):
            matvec_multiply([[1.0, float("inf")]], [1.0, 1.0])

    def test_ragged_row_message_reports_index_and_lengths(self):
        # The message should name the offending row and both lengths.
        A = [[1, 2, 3], [4, 5]]
        with self.assertRaises(ValueError) as caught:
            matvec_multiply(A, [1, 1, 1])
        message = str(caught.exception)
        self.assertIn("A[1]", message)
        self.assertIn("2", message)
        self.assertIn("3", message)

    def test_overflow_in_a_row_raises_overflow_error(self):
        # Overflow inside any row must not silently yield inf.
        with self.assertRaises(OverflowError):
            matvec_multiply([[1e308, 1e308]], [10.0, 10.0])

    def test_tuple_of_tuples_is_accepted(self):
        # Valid non-list sequence types must keep working.
        self.assertEqual(matvec_multiply(((1, 2), (3, 4)), (5, 6)), [17.0, 39.0])

    def test_vector_is_validated_once_not_once_per_row(self):
        # Validation must be O(m + n), not O(m * n): count reads of x's elements.
        class CountingList(list):
            def __init__(self, values):
                super().__init__(values)
                self.reads = 0

            def __getitem__(self, index):
                self.reads += 1
                return super().__getitem__(index)

        rows, cols = 50, 10
        A = [[1.0] * cols for _ in range(rows)]
        x = CountingList([1.0] * cols)
        matvec_multiply(A, x)

        # cols reads to validate x, plus cols reads per row in the arithmetic loop.
        # Re-validating x on every row would cost another rows * cols reads.
        self.assertEqual(x.reads, cols + rows * cols)

if __name__ == "__main__":
    unittest.main(verbosity=2)
