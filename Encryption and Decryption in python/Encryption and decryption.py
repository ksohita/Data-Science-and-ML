alphasmall='abcdefghijklmnopqrstuvwxyz'
alphacap='ABCDEFGHIJKLMNOPQRSTUVWXYZ'

key=4

def encrypt():
    message = input('Enter the message to encrypt:')
    print(message)
    newmessage = ''
    for character in message:
        if character in alphasmall:
                    pos = alphasmall.find(character)
                    newpos = (pos+key)%26
                    newchar=alphasmall[newpos]
                    newmessage+=newchar

        elif character in alphacap:
            pos = alphacap.find(character)
            newpos = (pos + key) % 26
            newchar = alphacap[newpos]
            newmessage += newchar
        else:
                    newmessage += character
    print('Encrypted message:')
    print(newmessage)

def decrypt():
    message = input('Enter the message to decrypt')
    print(message)
    newmessage = ''
    for character in message:
        if character in alphasmall:
                    pos = alphasmall.find(character)
                    newpos = (pos-key)%26
                    newchar=alphasmall[newpos]
                    newmessage+=newchar

        elif character in alphacap:
            pos = alphacap.find(character)
            newpos = (pos - key) % 26
            newchar = alphacap[newpos]
            newmessage += newchar
        else:
                    newmessage += character
    print('Decrypted message:')
    print(newmessage)


encrypt()
decrypt()




