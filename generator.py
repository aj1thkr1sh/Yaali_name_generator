from network import *

max_length = 12

def sample(category, start_letter = 'A'):
    with torch.no_grad():
        category_tensor = categoryTensor(category)
        input = inputTensor(start_letter)
        hidden = rnn.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

def samples(category, start_letters = 'ABC'):
    for start_letter in start_letters:
        print(sample(category, start_letter))

category = 'Female' #input('Enter category : ')

while True:
    starting_letter = input('Enter Starting letter : ')
    print(sample(category, starting_letter))

samples('Male', 'RET')
samples('Female', 'RETMA')
# samples('Spanish', 'SER')
# samples('Chinese', 'CER')