# Define User Schema for Response
from typing import List, Optional
from pydantic import BaseModel
from datetime import date, datetime
from ninja import Query, Schema, Form, File, FilterSchema

class UserSchema(Schema):
    first_name: str
    email: str

class ImageSchema(Schema):
    base64_image: str

class Unauthorized(Schema):
    detail:str

class MyTokenObtainPairOutSchema(Schema):
    refresh: str
    access: str
    user: UserSchema

# Start (User Registration)
class UserNotFound(BaseModel):
    error: str

class UserRegisterSchema(Schema):
    first_name: str
    last_name: str 
    username: str
    password: str
    email: str

class UserRegisterResponse(BaseModel):
    message: str
    user_id: int

class EmployeeRegistrationSchema(Schema):
    user_id: int
    employee_number: int
    first_name: str
    middle_name: Optional[str] = None  # This makes the field optional
    last_name: str
    contact_number: int

class EmployeeFacialRegistration(Schema):
    employee_number: str
    base64_image: str

class SuccessFaceRegistration(BaseModel):
    message: str

class EmployeeNotFound(BaseModel):
    message: str

class ErrorAtDecodingImage(BaseModel):
    message: str

class ErrorAtFaceRegistration(BaseModel):
    message: str

class EmployeeRegistrationResponse(BaseModel):
    message: str
    employee_number: int

#Start (Attendance Logging) Endpoints

class AttendanceEmployeeFilterSchema(FilterSchema):
    employee_number: Optional[int] = None
    date: Optional[datetime] = None

class ShiftRecordSchema(Schema):
    employee_number: int  # ID of the employee (Foreign Key reference)
    name: str
    date: date
    clock_in_at_am: Optional[datetime]
    clock_out_at_am: Optional[datetime]
    clock_in_at_pm: Optional[datetime]
    clock_out_at_pm: Optional[datetime]
    

class SuccessResponse(BaseModel):
    result: List[dict]

class ErrorResponse(BaseModel):
    error: List[str]

class EmployeeCheckInDetail(BaseModel):
    name: str
    employee_number: str
    profile_image_url: str
    message: str

class EmployeeCheckInResponse(BaseModel):
    result: List[EmployeeCheckInDetail]
    
class AlreadyCheckedInResponse(BaseModel):
    result: List[EmployeeCheckInDetail]

class StatusMessage(BaseModel):
    message: str

class SpoofingDetectedResponse(BaseModel):
    result: List[StatusMessage]

class NotFoundResponse(BaseModel):
    result: List[StatusMessage]

#Start (Attendance Logging) Endpoints
# Define User Schema for Response
class AttendanceEmployeeFilterSchema(FilterSchema):
    employee_number: Optional[int] = None
    date: Optional[datetime] = None

class ShiftRecordSchema(Schema):
    employee_number: int  # ID of the employee (Foreign Key reference)
    name: str
    date: date
    clock_in_at_am: Optional[datetime]
    clock_out_at_am: Optional[datetime]
    clock_in_at_pm: Optional[datetime]
    clock_out_at_pm: Optional[datetime]
    

class SuccessResponse(BaseModel):
    result: List[dict]

class ErrorResponse(BaseModel):
    error: List[str]

class EmployeeCheckInDetail(BaseModel):
    name: str
    employee_number: str
    profile_image_url: str
    message: str

class EmployeeCheckInResponse(BaseModel):
    result: List[EmployeeCheckInDetail]
    
class AlreadyCheckedInResponse(BaseModel):
    result: List[EmployeeCheckInDetail]

class StatusMessage(BaseModel):
    message: str

class SpoofingDetectedResponse(BaseModel):
    result: List[StatusMessage]

class NotFoundResponse(BaseModel):
    result: List[StatusMessage]