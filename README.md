# API de análise de retenção de clientes

SETUP

- Executar o comando `python main.py`
- Chamar o endpoint `/score` com autenticação basic Token (usuário: admin, senha: 123)
   - Body da requisição para cliente não assíduo
{
    "saw_sizecharts": 0,
    "saw_account_upgrade": 0,
    "detail_wishlist_add": 0,
    "saw_delivery": 0,
    "account_page_click": 0,
    "checked_returns_detail": 0,
    "device_mobile": 0,
    "promo_banner_click": 0,
    "device_tablet": 1,
    "loc_uk": 1,
    "sort_by": 1,
    "returning_user": 0,
    "image_picker": 0,
    "device_computer": 0,
    "closed_minibasket_click": 0,
    "saw_homepage": 1,
    "list_size_dropdown": 1,
    "basket_add_list": 1,
    "basket_add_detail": 0,
    "basket_icon_click": 0,
    "sign_in": 0
}

   - Body da requisição para cliente assíduo
{
    "saw_sizecharts": 0,
    "saw_account_upgrade": 0,
    "detail_wishlist_add": 0,
    "saw_delivery": 0,
    "account_page_click": 0,
    "checked_returns_detail": 0,
    "device_mobile": 1,
    "promo_banner_click": 0,
    "device_tablet": 0,
    "loc_uk": 1,
    "sort_by": 0,
    "returning_user": 0,
    "image_picker": 0,
    "device_computer": 0,
    "closed_minibasket_click": 0,
    "saw_homepage": 1,
    "list_size_dropdown": 1,
    "basket_add_list": 1,
    "basket_add_detail": 1,
    "basket_icon_click": 1,
    "sign_in": 1 
}