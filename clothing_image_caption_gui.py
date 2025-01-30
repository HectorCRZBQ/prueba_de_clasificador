import os
from tkinter import Tk, filedialog, Label, Button, Frame, Canvas
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel
import torch

# Cargar modelos de Hugging Face
def load_models():
    global blip_processor, blip_model, clip_processor, clip_model
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-vqa-base")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Definir categorías, subcategorías y estados
categories = {
    "Bañadores": [],
    "Calcetines y ropa interior": ["Albornoces", "Calcetines", "Calzoncillos y bóxers", "Pijamas"],
    "Camisas": [],
    "Jerséis y sudaderas": ["Cárdigans", "Jerséis", "Jerséis con cuello de pico", "Jerséis con cuello de tortuga", "Jerséis de cuello redondo", "Jerséis de punto", "Jerséis largos", "Sudaderas con cremallera", "Sudaderas con y sin capucha", "Otros jerséis"],
    "Pantalones": ["Chinos", "Joggers", "Pantalones anchos", "Pantalones cortos", "Pantalones de pinzas", "Pantalones pitillo", "Otros pantalones"],
    "Vaqueros": ["Vaqueros ajustados", "Vaqueros pitillo", "Vaqueros rectos", "Vaqueros rotos", "Otros vaqueros"],
    "Ropa de abrigo": ["Abrigos", "Capas y ponchos", "Chalecos y gilets", "Chaqueta bomber", "Chaquetas", "Chaquetas bolero", "Chubasqueros", "Gabardina", "Parkas"],
    "Ropa deportiva": ["Accesorios deportivos", "Chándales", "Pantalones", "Pantalones cortos", "Ropa de abrigo", "Sudaderas con y sin capucha", "Tops y camisetas", "Otra ropa deportiva", "Otras prendas exteriores"],
    "Tops y camisetas": ["Camisetas", "Chalecos y camisetas sin mangas", "Otros tops"],
    "Trajes y americanas": ["Chalecos", "Chaquetas de traje y americanas", "Conjuntos de traje", "Pantalones de traje", "Trajes de boda", "Otros trajes y americanas"],
    "Otras prendas": []
}

conditions = ["Sin estrenar (Conserva la etiqueta)", "Nuevo (Nunca se ha usado)", "Como nuevo (En perfectas condiciones)", "En buen estado (Bastante usado, pero bien conservado)", "En condiciones aceptables (Con evidentes signos de desgaste)", "Lo ha dado todo (Puede que toque repararlo)"]

# Seleccionar imagen
def select_image():
    global selected_image, img_tk
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")])
    if not file_path:
        return
    selected_image = file_path
    img = Image.open(selected_image)
    img.thumbnail((400, 400))
    img_tk = ImageTk.PhotoImage(img)
    img_canvas.delete("all")
    img_canvas.create_image(200, 200, image=img_tk, anchor="center")

# Procesar imagen con CLIP
def process_image(image, texts):
    image = image.convert("RGB")
    inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probabilities = logits_per_image.softmax(dim=1)
    return probabilities

# Generar descripción con BLIP
def generate_description():
    global result_label
    if not selected_image:
        result_label.config(text="Por favor, selecciona una imagen primero.")
        return
    try:
        image = Image.open(selected_image)
        image = image.convert("RGB")
        inputs = blip_processor(images=image, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        result_label.config(text=f"Descripción: {caption}")
    except Exception as e:
        result_label.config(text=f"Error al generar la descripción: {e}")

# Categorizar producto
def categorize_product():
    global result_label
    if not selected_image:
        result_label.config(text="Por favor, selecciona una imagen primero.")
        return
    try:
        image = Image.open(selected_image)

        # Añadir un contexto más claro a las categorías
        category_texts = []
        for category in categories.keys():
            category_texts.append(f"Imagen de {category.lower()}")

        # Comparar imagen con categorías
        category_probs = process_image(image, category_texts)
        main_category_idx = category_probs.argmax()
        main_category = list(categories.keys())[main_category_idx]
        
        if categories[main_category]:  # Si tiene subcategorías
            subcategory_texts = [f"Imagen de {subcategory.lower()}" for subcategory in categories[main_category]]
            subcategory_probs = process_image(image, subcategory_texts)
            subcategory_idx = subcategory_probs.argmax()
            subcategory = categories[main_category][subcategory_idx]
            result_label.config(text=f"Categoría: {main_category} -> {subcategory}")
        else:
            result_label.config(text=f"Categoría: {main_category}")
    except Exception as e:
        result_label.config(text=f"Error en la clasificación: {e}")

# Estimar estado
def estimate_condition():
    global result_label
    if not selected_image:
        result_label.config(text="Por favor, selecciona una imagen primero.")
        return
    try:
        image = Image.open(selected_image)
        condition_probs = process_image(image, conditions)
        predicted_condition = conditions[condition_probs.argmax()]
        result_label.config(text=f"Estado estimado: {predicted_condition}")
    except Exception as e:
        result_label.config(text=f"Error al estimar el estado: {e}")

# Estimar precio
def estimate_price():
    global result_label
    if not selected_image:
        result_label.config(text="Por favor, selecciona una imagen primero.")
        return
    try:
        image = Image.open(selected_image)

        # Primero categorizamos el producto
        category_texts = [f"Imagen de {category.lower()}" for category in categories.keys()]
        category_probs = process_image(image, category_texts)
        main_category_idx = category_probs.argmax()
        main_category = list(categories.keys())[main_category_idx]

        # Estimación de estado
        condition_probs = process_image(image, conditions)
        predicted_condition = conditions[condition_probs.argmax()]

        # Lógica simple para estimar el precio en función de la categoría y estado
        price = 10  # Precio base
        if main_category in ["Pantalones", "Vaqueros", "Camisas"]:
            price += 20
        if predicted_condition == "Nuevo (Nunca se ha usado)":
            price += 30
        elif predicted_condition == "Lo ha dado todo (Puede que toque repararlo)":
            price -= 10

        result_label.config(text=f"Precio estimado: {price}€")

    except Exception as e:
        result_label.config(text=f"Error al estimar el precio: {e}")

# Crear interfaz
tk_app = Tk()
tk_app.title("Clasificación de Ropa")
tk_app.geometry("500x700")
load_models()
selected_image = None
frame = Frame(tk_app, width=400, height=400, bg="gray")
frame.pack(pady=10)
img_canvas = Canvas(frame, width=400, height=400, bg="gray")
img_canvas.pack()
Button(tk_app, text="Seleccionar Imagen", command=select_image).pack(pady=10)
Button(tk_app, text="Generar Descripción", command=generate_description).pack(pady=10)
Button(tk_app, text="Categorizar Producto", command=categorize_product).pack(pady=10)
Button(tk_app, text="Estimar Estado", command=estimate_condition).pack(pady=10)
Button(tk_app, text="Estimar Precio", command=estimate_price).pack(pady=10)
result_label = Label(tk_app, text="", wraplength=400, justify="center", fg="blue")
result_label.pack(pady=10)
tk_app.mainloop()
