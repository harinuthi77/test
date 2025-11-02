"""
Python Code Generation Test
Test: Implement a binary search tree with insert, search, and delete operations
"""

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        """Insert a value into the BST"""
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)

    def search(self, value):
        """Search for a value in the BST"""
        return self._search_recursive(self.root, value)

    def _search_recursive(self, node, value):
        if node is None:
            return False
        if node.value == value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def inorder_traversal(self):
        """Return inorder traversal of the tree"""
        result = []
        self._inorder_recursive(self.root, result)
        return result

    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.value)
            self._inorder_recursive(node.right, result)

# Test the implementation
if __name__ == "__main__":
    bst = BinarySearchTree()
    values = [50, 30, 70, 20, 40, 60, 80]

    print("Inserting values:", values)
    for val in values:
        bst.insert(val)

    print("Inorder traversal:", bst.inorder_traversal())

    search_values = [40, 90]
    for val in search_values:
        found = bst.search(val)
        print(f"Search for {val}: {'Found' if found else 'Not found'}")
