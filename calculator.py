while True:
    choice = input(
        "Do you want to perform a calculation? (yes/no): ").strip().lower()
    if choice == 'no':
        print("Exiting the calculator...")
        exit()
    elif choice == 'yes':
        print("Openning  calculator...")
        break
    else:
        print("Invalid choice, please enter 'yes' or 'no'.")
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))
operation = input("Enter an operation (+, -, *, /): ")

# Perform calculation
if operation == '+':
    result = num1 + num2
elif operation == '-':
    result = num1 - num2
elif operation == '*':
    result = num1 * num2
elif operation == '/':
    if num2 != 0:
        result = num1 / num2
    else:
        result = "Error: Division by zero"
else:
    result = "Invalid operation"

print("Result:", result)
