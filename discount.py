def calculate_discount(price, discount_percent):
    if discount_percent >= 20:
        final_price = price - (price * discount_percent / 100)
        return final_price
    else:
        return price


price = int(input("Enter Price:"))
discount_percent = int(input("enter discount percent:"))

final_price = calculate_discount(price, discount_percent)
print("Final Price after discount is:",
      calculate_discount(price, discount_percent))

if discount_percent >= 20:
    print("Disount applied:")
else:
    print("No discount applied:")
# print("Final Price after discount is:", final_price)
print("Thank you for shopping with us!")
