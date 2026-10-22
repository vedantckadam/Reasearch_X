import tkinter as tk 

root = tk.Tk()

root.title("Additional Calculator")
root.geometry("400x300")

tk.Label(root,text="Marvellous Infosystems",font=("Arial",18,"bold")).pack(pady=10)

tk.Label(root,text="Enter First Number").pack()
entry1 = tk.Entry(root)
entry1.pack()

tk.Label(root,text="Enter Second Number").pack()
entry2 = tk.Entry(root)
entry2.pack()

def Addition():
    no1 = int(entry1.get())
    no2 = int(entry2.get())

    Result = no1 + no2

    labelResult.config(text = "Addition is :"+str(Result))


tk.Button(root,text="Additon",command=Addition).pack(pady=15)

labelResult=tk.Label(root,text="",font=("Arial",14))
labelResult.pack()

root.mainloop()



