import streamlit as st
from PIL import Image, ImageDraw, ImageFont

def main():
    st.set_page_config(page_title="Picture Description App", layout="wide")

    st.sidebar.title("Tabs")
    tabs = ["Picture", "About"]
    selected_tab = st.sidebar.selectbox("Select a tab", tabs)

    if selected_tab == "Picture":
        st.title("Picture")

        # Define the parts of the picture and their descriptions
        parts = {
            "Part A": "Description of Part A.",
            "Part B": "Description of Part B.",
            "Part C": "Description of Part C.",
        }

        # Load the picture
        picture_path = 'drilling_rig.JPG'  # Replace with the path to your picture
        picture = Image.open(picture_path)

        # Define the CSS style to control the size of the picture and the sidebar
        picture_style = f"width: 100%; max-width: 800px; height: auto;"
        sidebar_style = f"width: 200px;"

        # Apply the CSS style to the picture and the sidebar
        st.markdown(f'<img src="data:image/jpeg;base64,{image_to_base64(picture)}" style="{picture_style}">', unsafe_allow_html=True)
        st.sidebar.markdown(f'<div style="{sidebar_style}">', unsafe_allow_html=True)

        # Display the descriptions
        for part_name, description in parts.items():
            if st.sidebar.button(part_name):
                st.write(description)

        st.sidebar.markdown("</div>", unsafe_allow_html=True)

    else:
        st.title("About")
        st.write("This app was created by [Your Name] as a Streamlit exercise.")

def image_to_base64(image):
    from io import BytesIO
    import base64
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

if __name__ == "__main__":
    main()
