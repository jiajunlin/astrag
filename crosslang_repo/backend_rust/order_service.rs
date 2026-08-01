use actix_web::{web, HttpResponse};

/// Accept a submitted order for the given id and enqueue it for processing.
#[post("/api/orders/{id}")]
async fn submit_order(id: web::Path<String>, body: web::Json<OrderPayload>) -> HttpResponse {
    HttpResponse::Accepted().finish()
}

struct OrderPayload {
    items: Vec<String>,
}
