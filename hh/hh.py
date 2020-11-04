from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("anonymous-german-nlp/german-gpt2")

model = AutoModelWithLMHead.from_pretrained("anonymous-german-nlp/german-gpt2")

PADDING_TEXT = """Wer hat als kleines Kind nicht davon geträumt ein Autobett in Form eines Rennwagens oder Feuerwehrautos zu haben? Diesen Traum können Sie Ihren Kindern jetzt erfüllen! Möbel-Lux bietet eine große Auswahl an Autobetten – da ist für jeden das richtige Autobett dabei, egal ob Junge oder Mädchen.
In der Regel sind Kinder nicht begeistert abends ins Bett zu gehen. In der Regel ist ein Bett eben nur ein Bett. Nicht so mit einem Autobett. Kinder freuen sich regelrecht darauf in dem coolen Autobett zu liegen und so steigt auch die Lust daran, abends ins Bett zu gehen. Doch welches Modell ist das richtige für Sie und Ihr Kind?<eod> </s> <eos>"""
prompt = "Etagenbett"
inputs = tokenizer.encode(PADDING_TEXT + prompt, add_special_tokens=False, return_tensors="pt")
prompt_length = len(tokenizer.decode(inputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
outputs = model.generate(inputs, max_length=250, do_sample=True, top_p=0.96, top_k=60)
generated = tokenizer.decode(outputs[0])[prompt_length:]
print(prompt+generated)