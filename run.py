from utilities.generate_response import GenerateResponse


def main():
    gr = GenerateResponse()
    text = "input"

    while text != "":
        text = input("message : ")
        if text == "":
            break

        print("response : {}".format(gr(text)))
        print("-" * 10)


if __name__ == "__main__":
    main()
