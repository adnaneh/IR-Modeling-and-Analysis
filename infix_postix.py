# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:53:15 2020

@author: adnane
"""


class Conversion:

    # Constructor to initialize the class variables
    def __init__(self, capacity):
        self.top = -1
        self.capacity = capacity
        # This array is used a stack
        self.array = []
        # Precedence setting
        self.output = []
        self.precedence = {'OR': 1, 'NOT OR': 1, 'AND': 2, 'NOT AND': 2}

    # check if the stack is empty
    def isEmpty(self):
        return True if self.top == -1 else False

    # Return the value of the top of the stack
    def peek(self):
        return self.array[-1]

    # Pop the element from the stack
    def pop(self):
        if not self.isEmpty():
            self.top -= 1
            return self.array.pop()
        else:
            return "$"

    # Push the element to the stack
    def push(self, op):
        self.top += 1
        self.array.append(op)

    # A utility function to check is the given character
    # is operand
    def isOperand(self, ch):
        #        print((not ch in self.precedence))
        return (not ch in self.precedence)

    # Check if the precedence of operator is strictly
    # less than top of stack or not
    def notGreater(self, i):
        try:
            a = self.precedence[i]
            b = self.precedence[self.peek()]
            return True if a <= b else False
        except KeyError:
            return False

    # The main function that converts given infix expression
    # to postfix expression
    def infixToPostfix(self, exp):

        # Iterate over the expression for conversion
        for i in exp:
            # If the character is an operand,
            # add it to output
            if self.isOperand(i):
                self.output.append(i)

            # If the character is an '(', push it to stack
            elif i == '(':
                self.push(i)

            # If the scanned character is an ')', pop and
            # output from the stack until and '(' is found
            elif i == ')':
                while((not self.isEmpty()) and self.peek() != '('):
                    a = self.pop()
                    self.output.append(a)
                if (not self.isEmpty() and self.peek() != '('):
                    return -1
                else:
                    self.pop()

            # An operator is encountered
            else:
                while(not self.isEmpty() and self.notGreater(i)):
                    self.output.append(self.pop())
                self.push(i)

        # pop all the operator from the stack
        while not self.isEmpty():
            self.output.append(self.pop())

        return(self.output)

# Driver program to test above function


def transformation_query_to_postfixe(query):

    processed_query = []
    i = 0
    while i < len(query):
        if i < len(query) - 1:
            s1 = query[i]
            s2 = query[i + 1]
            if s1 == 'AND' and s2 == 'NOT':
                processed_query.append('NOT AND')
                i += 2
                continue
            if s1 == 'OR' and s2 == 'NOT':
                processed_query.append('NOT OR')
                i += 2
                continue
        processed_query.append(query[i])
        i += 1
    obj = Conversion(len(query))
    res = obj.infixToPostfix(query)

    processed_res = []
    i = 0

    while i < len(res):
        s = res[i]
        if s == 'NOT AND':
            processed_res += ['NOT', 'AND']
            continue
        if s == ('NOT OR'):
            processed_res += ['NOT', 'OR']
            continue
        processed_res.append(res[i])
        i += 1
    return(processed_res)
