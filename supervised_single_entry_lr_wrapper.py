from typing import List, Dict, Set, Tuple

def ProperSuffix(u: str, s: str) -> bool:
    """
    Determines if u is a proper suffix of s.
    u is a proper suffix of s if u is a suffix of s and only occurs as a suffix
    (i.e., s ends with u and u does not occur anywhere else in s)

    Example:
    deabc is a proper suffix of deabcde but de is not because it occurs as a prefix as well.

    Args:
        u: string to check
        s: string to check

    Returns:
        True if u is a proper suffix of s, False otherwise
    """
    return s.endswith(u) and u not in s[:-len(u)]

class SingleEntryLRWrapper:
    """
    Left-Right Wrapper for documents with a single entry with multiple attributes
    The documents are of the form where the info we are trying to extract is of the form
    (Name, Age, Address, Phone, Email, etc.) and there will be only one instance of each in the text.

    Implementation of the LR wrapper algorithm as described in the paper:
    "Wrapper induction: Efficiency and expressiveness", 2000 by Kushmerick, et al. 
    """
    def __init__(self, examples_file_paths: List[str], labels: List[List[Tuple]]) -> None:
        """
        Args:
            examples_file_paths: list of file paths to get documents
            labels: list of lists representing each document. Sorted by left delimiter index (ascending).
            The i-th list contains beginning and ending indices for each attribute in the i-th example document.

        Initializes the wrapper (detemrines valid left and right delimiters for each attribute)
        """
        self.labels = labels
        if len(labels) == 0:
            raise ValueError("No training examples provided")
        
        self.examples = []
        print("Loading documents...")
        for path in examples_file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                self.examples.append(f.read())
        
        self.num_attributes = len(labels[0])
        self.left = [''] * self.num_attributes
        self.right = [''] * self.num_attributes

        print("Creating Wrapper...")
        print("Finding left delimiters...")
        self.GetValidLeft()
        print("Finding right delimiters...")
        self.GetValidRight()

    def GetValidLeft(self):
        """
        Gets the valid left delimiter for each attribute
        """
        for k in range(self.num_attributes):
            for u in self.LeftCandidates(k):
                if self.IsValidLeft(u, k):
                    self.left[k] = u
                    break

    def GetValidRight(self):
        """
        Gets the valid right delimiter for each attribute
        """
        for k in range(self.num_attributes):
            for u in self.RightCandidates(k):
                if self.IsValidRight(u, k):
                    self.right[k] = u
                    break

    def LeftCandidates(self, k):
        """
        Returns a set of possible left delimiters for the k-th attribute.
        Candidates are the suffixes of the shortest string occurring to the
        left of the k-th attribute in each example.

        Args:
            k: index of the attribute to find candidates for

        Returns:
            set of possible left delimiters for the k-th attribute
        """
        # get shortest left neighbor of k-th attribute across all examples
        n = self.LeftNeighbors(k)
        shortest = min(n, key=len)
        # get suffixes of shortest
        suffixes = {shortest[i:] for i in range(len(shortest))} # set
        return suffixes

    def RightCandidates(self, k):
        """
        Returns a set of possible right delimiters for the k-th attribute.
        Candidates are the prefixes of the shortest string occurring to the
        right of the k-th attribute in each example.

        Args:
            k: index of the attribute to find candidates for

        Returns:
            set of possible right delimiters for the k-th attribute
        """
        # get shortest right neighbor of k-th attribute across all examples
        n = self.RightNeighbors(k)
        shortest = min(n, key=len)
        # get prefixes of shortest
        prefixes = {shortest[:i] for i in range(len(shortest))} # set
        return prefixes
    
    def IsValidLeft(self, u, k):
        """
        Determines if u is a valid left delimiter for the k-th attribute.

        Validity is determined by the following conditions:
        1. u must be a *proper suffix* of the text that occurs immediately before
        each instance of attribute k in every example page.
        2. for the 1st attr only, u must not be a substring of any example page's tail.

        Args:
            u: left delimiter to check
            k: index of the attribute to check
        
        Returns:
            True if u is a valid left delimiter for the k-th attribute, False otherwise
        """
        for s in self.LeftNeighbors(k):
            if not ProperSuffix(u, s):
                return False
        if k == 0:
            for s in self.example_tails():
                if u in s:
                    return False
        return True

    def IsValidRight(self, u, k):
        """
        Determines if u is a valid right delimiter for the k-th attribute.

        Validity is determined by the following conditions:
        1. u must not be a substring of any instance of attribute k in any of the example pages.
        2. u must be a prefix of the text that occurs immediately following the k-th attribute k in every example page.

        Args:
            u: right delimiter to check
            k: index of the attribute to check

        Returns:
            True if u is a valid right delimiter for the k-th attribute, False otherwise
        """
        for s in self.attributes(k):
            if u in s:
                return False
        for s in self.RightNeighbors(k):
            if not s.startswith(u):
                return False
        return True
    
    def attributes(self, k):
        """
        Returns a set of the text of all instances of the k-th attribute in all example pages.

        Args:
            k: index of the attribute to get instances of

        Returns:
            set of text all instances of the k-th attribute in all example pages
        """
        return {P[self.labels[i][k][0]:self.labels[i][k][1]] for (i, P) in enumerate(self.examples)} 

    def LeftNeighbors(self, k):
        """
        Returns a set of all the strings that occur immediately before the k-th attribute
        and after the k-1-th attribute in each example page.

        Args:
            k: index of the attribute to get left neighbors of

        Returns:
            set of all the strings that occur between the k-1-th and k-th attributes in each example page
        """
        if k == 0:
            return self.Seps(self.num_attributes-1).union(self.example_heads())
        return self.Seps(k-1)
    
    def RightNeighbors(self, k):
        """
        Returns a set of all the strings that occur immediately after the k-th attribute
        and before the k+1-th attribute in each example page.

        Args:
            k: index of the attribute to get right neighbors of

        Returns:
            set of all the strings that occur between the k-th and k+1-th attributes in each example page
        """
        if k == self.num_attributes - 1:
            return self.Seps(self.num_attributes-1).union(self.example_tails())
        return self.Seps(k)
    
    def example_heads(self):
        """
        Returns a set of all the strings that occur at the beginning of each example page before the 1st attribute.
        """
        return {P[:self.labels[i][0][0]] for (i, P) in enumerate(self.examples)}
    
    def example_tails(self):
        """
        Returns a set of all the strings that occur at the end of each example page after the last attribute.
        """
        return {P[self.labels[i][-1][1]:] for (i, P) in enumerate(self.examples)}
    
    def Seps(self, k):
        """
        Returns a set of all the strings that occur between the k-th and (k mod K)+1-th attributes in each example page.
        """
        if k == self.num_attributes - 1:
            return set()
        return {P[self.labels[i][k][1]:self.labels[i][k+1][0]] for (i, P) in enumerate(self.examples)}
    
    def execLR(self, P):
        """
        Executes the left and right delimiter search on the given page to extract fields (i.e., runs the wrapper on the input page).

        Args:
            P: A given page to apply wrapper to

        Returns:
            Attributes extracted using the left and right delimiters.
        """
        attr = []
        for (l, r) in zip(self.left, self.right):
            # extract text between l and r
            l_idx = P.find(l)+len(l)
            r_idx = P[l_idx:].find(r) + l_idx
            attr.append(P[l_idx:r_idx])
            P = P[r_idx:] # update P to make search more efficient
        return attr
